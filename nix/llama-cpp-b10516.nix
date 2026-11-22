{ lib
, stdenv
, fetchFromGitHub
, cmake
, ninja
, pkg-config
, openssl
, autoPatchelfHook
, vulkan-headers ? null
, vulkan-loader ? null
, shaderc ? null
, vulkan ? false
, hash
}:

stdenv.mkDerivation {
  pname = "llama-cpp-b10516${lib.optionalString vulkan "-vulkan"}";
  version = "b10516";

  src = fetchFromGitHub {
    owner = "ggml-org";
    repo = "llama.cpp";
    rev = "b95502ba9aa0eb73a2f4fc8878d7fbe6a847a0b9";
    hash = hash;
  };

  nativeBuildInputs = [ cmake ninja pkg-config autoPatchelfHook ]
    ++ lib.optionals vulkan [ shaderc ];
  buildInputs = [ openssl ]
    ++ lib.optionals vulkan [ vulkan-headers vulkan-loader shaderc ];

  # Flags follow b10516 .github/workflows/release.yml except the two Nix
  # deviations: no CPU variant fan-out, and no RPC. Do not set
  # CMAKE_INSTALL_RPATH='$ORIGIN'; that drops the store rpath stdenv writes
  # and vulkan-loader is then missing at runtime.
  cmakeFlags = [
    "-DCMAKE_BUILD_TYPE=Release"
    "-DGGML_NATIVE=OFF"
    "-DGGML_CPU_ALL_VARIANTS=OFF"
    "-DGGML_BACKEND_DL=ON"
    "-DGGML_RPC=OFF"
    "-DGGML_VULKAN=${if vulkan then "ON" else "OFF"}"
    "-DLLAMA_BUILD_EXAMPLES=OFF"
    "-DLLAMA_BUILD_TESTS=OFF"
    "-DLLAMA_BUILD_TOOLS=ON"
    "-DLLAMA_BUILD_SERVER=ON"
    # CMakeLists.txt: option(LLAMA_BUILD_UI ... ON). The prebuilt UI fetch
    # needs network, and there is an OFF switch, so do not add nodejs.
    "-DLLAMA_BUILD_UI=OFF"
    # tools/ui still provisions assets when LLAMA_USE_PREBUILT_UI stays ON.
    "-DLLAMA_USE_PREBUILT_UI=OFF"
    # CMake's build RPATH points at /build. auditTmpdir runs before
    # autoPatchelfHook, so the build tree must not be in the binary at all.
    "-DCMAKE_SKIP_BUILD_RPATH=ON"
  ];

  installPhase = ''
    runHook preInstall
    mkdir -p "$out"
    # cmakeConfigurePhase may already be inside ./build. The builder is not
    # bash, so do not use compgen; nullglob also drops unmatched globs.
    if [ -x build/bin/llama-server ]; then
      binDir=build/bin
      libDir=build/lib
    elif [ -x bin/llama-server ]; then
      binDir=bin
      libDir=lib
    else
      echo "llama-server was not produced in build/bin or bin" >&2
      exit 1
    fi
    cp -a "$binDir/llama-server" "$out/"
    find "$binDir" -maxdepth 1 -name '*.so*' -exec cp -a {} "$out/" \;
    if ! find "$out" -maxdepth 1 -name 'libggml*.so' -print -quit | grep -q .; then
      find "$libDir" -maxdepth 1 \( -name 'libggml*.so*' -o -name 'libllama*.so*' -o -name 'libmtmd*.so*' \) -exec cp -a {} "$out/" \;
    fi
    test -x "$out/llama-server"
    if ! find "$out" -maxdepth 1 -name 'libggml*.so' -print -quit | grep -q .; then
      echo "llama-server was built without a libggml shared library" >&2
      exit 1
    fi
    ${lib.optionalString vulkan ''
      if [ ! -e "$out/libggml-vulkan.so" ]; then
        echo "vulkan sidecar is missing libggml-vulkan.so" >&2
        exit 1
      fi
    ''}
    runHook postInstall
  '';
  # autoPatchelfHook (postFixup) rewrites RPATH to sibling libraries in $out
  # and to openssl / vulkan-loader. Do not set CMAKE_INSTALL_RPATH='$ORIGIN'.
}
