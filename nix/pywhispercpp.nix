{ lib, python3Packages, fetchFromGitHub, cmake, ninja }:
let
  installer = builtins.readFile ../lib/src/backend_installer.py;
  # Matching .* over the whole installer exhausts the regex engine's stack
  # on runners with an 8 MiB limit. Match only the selected assignment line.
  constant = name:
    let
      lines = builtins.filter (lib.hasPrefix "${name} = ") (lib.splitString "\n" installer);
      value = if builtins.length lines == 1
        then builtins.match ''${name} = "([^"]+)"'' (builtins.head lines)
        else null;
    in if value == null then throw "Expected one quoted ${name} assignment in backend_installer.py"
      else builtins.head value;
  version = constant "PYWHISPERCPP_VERSION";
  revision = constant "PYWHISPERCPP_PINNED_COMMIT";
in
assert lib.hasInfix "pywhispercpp==${version}\n" (builtins.readFile ../requirements-pywhispercpp.txt);
python3Packages.buildPythonPackage {
  pname = "pywhispercpp";
  inherit version;
  pyproject = true;
  src = fetchFromGitHub {
    owner = "Absadiki";
    repo = "pywhispercpp";
    rev = revision;
    fetchSubmodules = true;
    hash = "sha256-FIYlAPkvuSGGp+XpveGVc9uZx/ND008k8PPsTHGmadA=";
  };
  nativeBuildInputs = [ cmake ninja ];
  build-system = [
    python3Packages.setuptools python3Packages.wheel python3Packages.setuptools-scm
    python3Packages.cmake python3Packages.ninja
  ];
  postPatch = ''
    # Upstream only invokes repairwheel on Windows; Nix fixes ELF references.
    substituteInPlace pyproject.toml --replace-fail '"repairwheel",' ""
  '';
  dependencies = with python3Packages; [ numpy requests tqdm platformdirs ];
  dontUseCmakeConfigure = true;
  env = {
    PYWHISPERCPP_VERSION = version;
    CMAKE_ARGS = lib.concatStringsSep " " [
      "-DGGML_NATIVE=OFF" "-DGGML_OPENMP=OFF" "-DGGML_CUDA=OFF"
      "-DGGML_VULKAN=OFF" "-DGGML_HIP=OFF" "-DGGML_BLAS=OFF"
      "-DBUILD_SHARED_LIBS=OFF" "-DWHISPER_BUILD_TESTS=OFF"
      "-DWHISPER_BUILD_EXAMPLES=OFF" "-DFETCHCONTENT_FULLY_DISCONNECTED=ON"
    ];
  };
  preBuild = ''
    export CMAKE_BUILD_PARALLEL_LEVEL="$NIX_BUILD_CORES"
  '';
  # Upstream tests require downloaded models. Importing the native extension
  # checks linking without opening audio devices or accessing the network.
  doCheck = false;
  pythonImportsCheck = [ "_pywhispercpp" "pywhispercpp.model" ];
  meta = {
    description = "Python bindings for whisper.cpp, CPU-only";
    homepage = "https://github.com/Absadiki/pywhispercpp";
    license = lib.licenses.mit;
    platforms = [ "x86_64-linux" ];
  };
}
