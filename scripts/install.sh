#!/usr/bin/env bash
#
# hyprwhspr bootstrap installer
#
# Usage:
#   curl -fsSL https://hyprwhspr.com/install.sh | bash
#
# Clones (or updates) hyprwhspr to a managed location, installs distro
# dependencies, then runs interactive setup. Re-running is the update path.

set -euo pipefail

# Rollout gate: enable the release default only after a compatible tag is published
# and desktop VM acceptance passes. Explicit lifecycle flags and existing managed
# installs already select the release path, without repeating onboarding.
release_bootstrap() {
if [[ -f /etc/os-release ]]; then
    . /etc/os-release
    if [[ "${ID:-}" == arch || " ${ID_LIKE:-} " == *" arch "* ]]; then
        echo 'Use your AUR package manager to install or update hyprwhspr, then run hyprwhspr setup.'
        exit 0
    fi
    case " ${ID:-} ${ID_LIKE:-} " in
        *debian*|*ubuntu*|*fedora*|*rhel*|*suse*) ;;
        *) echo "Unsupported bootstrap distro: ${ID:-unknown}" >&2; exit 1 ;;
    esac
fi
# Release bootstrap. Kept opt-in until the first compatible release is published.
set -euo pipefail
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
python_path=''
explicit_python=''
version=''
repair=0
while (($#)); do
    case "$1" in
        --python|--version)
            (($# >= 2)) || { echo "$1 requires a value" >&2; exit 1; }
            if [[ "$1" == --python ]]; then python_path="$2"; explicit_python="$2"; else version="$2"; fi
            shift 2 ;;
        --repair) repair=1; shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done
if [[ -z "$python_path" ]]; then
    for candidate in /usr/bin/python3 /usr/local/bin/python3; do
        if [[ -x "$candidate" ]]; then python_path="$candidate"; break; fi
    done
fi
[[ -n "$python_path" ]] || { echo 'Install Python 3.11–3.14 or pass --python PATH.' >&2; exit 1; }
exec "$python_path" -I -B - "$version" "$python_path" "$repair" "$explicit_python" <<'PY'
import hashlib,json,os,pathlib,re,subprocess,sys,tempfile,urllib.request,urllib.error
version,python,repair,explicit_python=sys.argv[1:]
if repair=='1' and not version:
    current=pathlib.Path(os.environ.get('XDG_DATA_HOME',str(pathlib.Path.home()/'.local/share')))/'hyprwhspr/current.json'
    if current.exists():
        try:
            installed=json.loads(current.read_text(encoding='utf-8'))['version']
            if not isinstance(installed,str) or not re.fullmatch(r'v?\d+\.\d+\.\d+',installed): raise ValueError('invalid installed version')
            version=installed
        except (OSError,ValueError,KeyError,TypeError) as exc:
            print(f'Cannot read installed version from {current}: {exc}. Repair will select the latest compatible release.',file=sys.stderr)
if version and not re.fullmatch(r'v?\d+\.\d+\.\d+',version): sys.exit('Invalid application version')
base='https://api.github.com/repos/goodroot/hyprwhspr/releases'
base+=('/tags/v'+version.lstrip('v')) if version else '?per_page=100'
class HTTPSOnly(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        if not newurl.startswith('https://'): sys.exit(f'Refusing plaintext redirect to {newurl}. Release downloads require HTTPS.')
        return super().redirect_request(req,fp,code,msg,headers,newurl)
urllib.request.install_opener(urllib.request.build_opener(HTTPSOnly))
def fetch(url, limit):
    if not url.startswith('https://'): sys.exit(f'Release downloads require HTTPS: {url}')
    try:
        with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'hyprwhspr-bootstrap'}),timeout=60) as response:
            content=response.read(limit+1)
    except urllib.error.HTTPError as exc:
        sys.exit(f'Release download failed (HTTP {exc.code}): {url}. Check the requested version or retry after the GitHub rate limit resets.')
    except (urllib.error.URLError,OSError) as exc:
        sys.exit(f'Release download failed: {url}: {exc}. Check connectivity and retry.')
    if len(content)>limit: raise ValueError('Download exceeds size limit')
    return content
releases=json.loads(fetch(base,8*1024*1024))
if version: releases=[releases]
candidates=[r for r in releases if not r['draft'] and not r['prerelease'] and re.fullmatch(r'v?\d+\.\d+\.\d+',r['tag_name']) and (not version or r['tag_name'].lstrip('v')==version.lstrip('v')) and {'managed_install.py','SHA256SUMS'}.issubset({a['name'] for a in r['assets']})]
if not candidates: sys.exit('No compatible published application release found.')
release=max(candidates,key=lambda r:tuple(map(int,r['tag_name'].lstrip('v').split('.'))))
assets={a['name']:a['browser_download_url'] for a in release['assets']}
checks=fetch(assets['SHA256SUMS'],65536).decode().splitlines()
expected=next((parts[0] for line in checks if len(parts:=line.split())==2 and parts[1]=='managed_install.py'),None)
if expected is None or not re.fullmatch(r'[0-9a-fA-F]{64}',expected): sys.exit('Release SHA256SUMS has no valid managed_install.py checksum.')
content=fetch(assets['managed_install.py'],1024*1024)
if hashlib.sha256(content).hexdigest()!=expected: sys.exit('Recovery helper checksum mismatch')
with tempfile.TemporaryDirectory(prefix='hyprwhspr-bootstrap-') as temporary:
    helper=pathlib.Path(temporary)/'managed_install.py'
    helper.write_bytes(content)
    command=[python,'-I','-B',str(helper),'bootstrap','--version',release['tag_name']]
    if explicit_python: command.extend(['--python',explicit_python])
    if repair=='1': command.append('--repair')
    sys.exit(subprocess.call(command))
PY

}
if [[ "${HYPRWHSPR_RELEASE_BOOTSTRAP:-0}" == 1 || $# -gt 0 || -f "${XDG_DATA_HOME:-$HOME/.local/share}/hyprwhspr/current.json" ]]; then
    release_bootstrap "$@"
fi

REPO_URL="https://github.com/goodroot/hyprwhspr.git"
CLONE_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/hyprwhspr/src"

BLUE='\033[0;34m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${BLUE}[INFO]${NC} $1"; }
ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
die()  { echo -e "${RED}[ERROR]${NC} $1" >&2; exit 1; }

echo ""
echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  hyprwhspr installer${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

if [[ -f /etc/os-release ]]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    if [[ "${ID:-}" == "arch" || " ${ID_LIKE:-} " == *" arch "* ]]; then
        log "Arch-based system detected — hyprwhspr is on the AUR:"
        echo ""
        echo "  yay -S hyprwhspr        # stable"
        echo "  yay -S hyprwhspr-git    # bleeding edge"
        echo ""
        echo "Then run: hyprwhspr setup"
        exit 0
    fi
    # Pre-flight before cloning; install-deps.sh remains the authority
    case " ${ID:-} ${ID_LIKE:-} " in
        *debian*|*ubuntu*|*fedora*|*rhel*|*suse*) ;;
        *) die "Unsupported distro '${ID:-unknown}' — supported: Ubuntu, Debian, Fedora, openSUSE (and derivatives).
See manual install: https://github.com/goodroot/hyprwhspr#other-linux-distros" ;;
    esac
fi

{ : </dev/tty; } 2>/dev/null || die "This installer is interactive and needs a terminal.
Run it from a shell: curl -fsSL https://hyprwhspr.com/install.sh | bash"

command -v git >/dev/null 2>&1 || die "git is required — install it with your package manager and re-run."

if [[ -d "$CLONE_DIR/.git" ]]; then
    log "Existing install found — updating $CLONE_DIR"
    git -C "$CLONE_DIR" pull --ff-only || die "Update failed — resolve manually in $CLONE_DIR and re-run."
else
    log "Cloning hyprwhspr to $CLONE_DIR"
    mkdir -p "$(dirname "$CLONE_DIR")"
    git clone "$REPO_URL" "$CLONE_DIR"
fi
ok "Source ready at $CLONE_DIR"

log "Installing distro dependencies..."
bash "$CLONE_DIR/scripts/install-deps.sh"

log "Starting interactive setup..."
"$CLONE_DIR/bin/hyprwhspr" setup </dev/tty

echo ""
ok "Done!"
echo "  Installed at:  $CLONE_DIR"
echo "  Update:        re-run this installer"
echo "  Uninstall:     hyprwhspr uninstall"
echo ""
