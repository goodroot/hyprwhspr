{
  description = "hyprwhspr Nix support (beta, Whisper.cpp CPU by default)";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      mkPackage = extraBackends: pkgs.callPackage ./nix/package.nix {
        inherit extraBackends;
        version = "0-unstable-${self.shortRev or self.dirtyShortRev or "local"}";
      };
      package = mkPackage [ ];
      qwenPackage = mkPackage [ "qwen3-asr" ];
      nixosModule = import ./nix/module.nix {
        inherit mkPackage;
        extraBackendNames = builtins.attrNames (import ./nix/backends.nix { inherit pkgs; });
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
