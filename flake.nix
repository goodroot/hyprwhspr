{
  description = "hyprwhspr Nix support (beta, Whisper.cpp CPU by default)";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";
  inputs.nixpkgs-llama.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs, nixpkgs-llama }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      llamaPkgs = import nixpkgs-llama { inherit system; };
      backends = import ./nix/backends.nix { inherit pkgs llamaPkgs; };
      mkPackage = extraBackends: pkgs.callPackage ./nix/package.nix {
        inherit backends extraBackends;
        version = "0-unstable-${self.shortRev or self.dirtyShortRev or "local"}";
      };
      package = mkPackage [ ];
      qwenPackage = mkPackage [ "qwen3-asr" ];
      nixosModule = import ./nix/module.nix {
        inherit mkPackage;
        extraBackendNames = builtins.attrNames backends;
      };
    in {
      packages.${system} = {
        hyprwhspr = package;
        hyprwhspr-qwen = qwenPackage;
        default = package;
      };
      apps.${system} = rec {
        hyprwhspr = { type = "app"; program = "${package}/bin/hyprwhspr"; };
        default = hyprwhspr;
      };
      nixosModules.default = nixosModule;
      lib.mkPackage = mkPackage;
      checks.${system} = import ./nix/checks.nix {
        inherit pkgs package nixpkgs nixosModule system;
      };
    };
}
