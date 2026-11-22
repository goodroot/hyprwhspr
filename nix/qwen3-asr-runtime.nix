{ stdenvNoCC, llama-cpp, llama-cpp-vulkan }:
stdenvNoCC.mkDerivation {
  pname = "hyprwhspr-qwen3-asr-runtime";
  version = llama-cpp.version;
  dontUnpack = true;
  installPhase = ''
    runHook preInstall
    mkdir -p "$out"
    ln -s ${llama-cpp} "$out/cpu"
    ln -s ${llama-cpp-vulkan} "$out/vulkan"
    # hyprwhspr status reports this instead of the pinned Ubuntu release.
    echo '${llama-cpp.version}' > "$out/version"
    runHook postInstall
  '';
}
