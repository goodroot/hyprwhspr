{ defaultPackage }:
{ config, lib, pkgs, ... }:
let
  cfg = config.services.hyprwhspr;
  # Reuse the canonical readiness and cleanup commands; resolve their shell
  # and PATH through Nix rather than copying another service implementation.
  template = lib.splitString "\n" (builtins.readFile ../config/systemd/hyprwhspr.service);
  directive = name: lib.removePrefix "${name}=" (lib.findFirst (lib.hasPrefix "${name}=") "" template);
  nixShell = command: builtins.replaceStrings [ "/bin/bash" " -lc " ] [ "${pkgs.bash}/bin/bash" " -c " ] command;
in {
  options.services.hyprwhspr = {
    enable = lib.mkEnableOption "hyprwhspr CPU dictation in graphical user sessions (beta)";
    package = lib.mkOption {
      type = lib.types.package;
      default = defaultPackage;
      description = "The hyprwhspr Nix package to run.";
    };
    users = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [];
      example = [ "alice" ];
      description = "Existing users allowed to run the service and access input devices.";
    };
  };
  config = lib.mkIf cfg.enable {
    assertions = [
      { assertion = cfg.users != []; message = "services.hyprwhspr.users must contain at least one existing user."; }
      { assertion = builtins.all (user: builtins.hasAttr user config.users.users &&
          (config.users.users.${user}.isNormalUser || config.users.users.${user}.isSystemUser)) cfg.users;
        message = "services.hyprwhspr.users must name users declared in users.users."; }
      { assertion = pkgs.stdenv.hostPlatform.system == "x86_64-linux";
        message = "The hyprwhspr Nix package currently supports x86_64-linux only."; }
    ];
    environment.systemPackages = [ cfg.package ];
    boot.kernelModules = [ "uinput" ];
    services.udev.extraRules = ''
      KERNEL=="uinput", GROUP="input", MODE="0660"
    '';
    users.groups.input = {};
    users.users = lib.genAttrs cfg.users (_: { extraGroups = [ "input" ]; });
    systemd.user.services.hyprwhspr = {
      description = "hyprwhspr stt";
      documentation = [ "https://github.com/goodroot/hyprwhspr" ];
      partOf = [ "graphical-session.target" ];
      after = [ "graphical-session.target" "pipewire.service" "wireplumber.service" ];
      wants = [ "pipewire.service" "wireplumber.service" ];
      wantedBy = [ "graphical-session.target" ];
      unitConfig = {
        ConditionUser = map (user: "|${user}") cfg.users;
        ConditionPathExists = "%E/hyprwhspr/config.json";
      };
      # Preserve host/user compositor tools without packaging a compositor.
      path = [ pkgs.coreutils pkgs.procps
        "/run/current-system/sw" "/etc/profiles/per-user/%u" "%h/.nix-profile"
      ];
      environment.PYTHONUNBUFFERED = "1";
      serviceConfig = {
        Type = "simple";
        ExecStartPre = nixShell (directive "ExecStartPre");
        ExecStart = lib.getExe cfg.package;
        ExecStopPost = nixShell (directive "ExecStopPost");
        Restart = "on-failure";
        RestartSec = 2;
        StandardOutput = "journal";
        StandardError = "journal";
      };
    };
  };
}
