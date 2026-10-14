{
  description = "hyprwhspr Nix support (beta, Whisper.cpp CPU)";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      package = pkgs.callPackage ./nix/package.nix {
        version = "0-unstable-${self.shortRev or self.dirtyShortRev or "local"}";
      };
      nixosModule = import ./nix/module.nix { defaultPackage = package; };
    in {
      packages.${system} = { hyprwhspr = package; default = package; };
      apps.${system} = rec {
        hyprwhspr = { type = "app"; program = "${package}/bin/hyprwhspr"; };
        default = hyprwhspr;
      };
      nixosModules.default = nixosModule;
      checks.${system} = import ./nix/checks.nix {
        inherit pkgs package nixpkgs nixosModule system;
      };
    };
}
