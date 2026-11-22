{ lib, stdenvNoCC, callPackage }:
let
  hash = "sha256-BHdG/RIoTvxT2YxU7Tg5/QQyHq1+dycaMCcf+hDhaCY=";
  cpu = callPackage ./llama-cpp-b10516.nix { inherit hash; vulkan = false; };
  vulkan = callPackage ./llama-cpp-b10516.nix { inherit hash; vulkan = true; };
in
assert lib.hasInfix ''LLAMA_CPP_RELEASE = "b10516"''
  (builtins.readFile ../lib/src/qwen3_asr_runtime.py);
stdenvNoCC.mkDerivation {
  pname = "hyprwhspr-qwen3-asr-runtime";
  version = "b10516";
  dontUnpack = true;
  installPhase = ''
    runHook preInstall
    mkdir -p "$out/b10516"
    ln -s ${cpu} "$out/b10516/cpu"
    ln -s ${vulkan} "$out/b10516/vulkan"
    runHook postInstall
  '';
}
