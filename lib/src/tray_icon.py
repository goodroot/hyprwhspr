"""
System tray icon for hyprwhspr
Publishes a StatusNotifierItem and its dbusmenu on the session bus, so any
standard tray (KDE Plasma, XFCE, Waybar's tray, Quickshell shells) shows it
"""

import os
import threading

try:
    import gi
    gi.require_version('Gio', '2.0')
    from gi.repository import Gio, GLib
    GIO_AVAILABLE = True
except (ImportError, ValueError):
    GIO_AVAILABLE = False


WATCHER_NAME = 'org.kde.StatusNotifierWatcher'
WATCHER_PATH = '/StatusNotifierWatcher'
ITEM_PATH = '/StatusNotifierItem'
MENU_PATH = '/MenuBar'
ITEM_IFACE = 'org.kde.StatusNotifierItem'
MENU_IFACE = 'com.canonical.dbusmenu'

POLL_INTERVAL_MS = 250

# state -> (theme icon name, label)
STATES = {
    'ready': ('audio-input-microphone', 'Ready'),
    'recording': ('media-record', 'Recording'),
    'loading': ('microphone-sensitivity-muted', 'Loading model'),
    'unloaded': ('microphone-sensitivity-muted', 'Model unloaded'),
    'error': ('dialog-warning', 'Error'),
}

# Menu item ids; 0 is the dbusmenu root
ITEM_STATUS = 1
ITEM_RECORD = 3
ITEM_MODEL = 4
ITEM_RESTART = 6
ITEM_STOP = 7

ITEM_XML = f"""
<node>
  <interface name="{ITEM_IFACE}">
    <property name="Category" type="s" access="read"/>
    <property name="Id" type="s" access="read"/>
    <property name="Title" type="s" access="read"/>
    <property name="Status" type="s" access="read"/>
    <property name="WindowId" type="i" access="read"/>
    <property name="IconName" type="s" access="read"/>
    <property name="IconThemePath" type="s" access="read"/>
    <property name="IconPixmap" type="a(iiay)" access="read"/>
    <property name="OverlayIconName" type="s" access="read"/>
    <property name="OverlayIconPixmap" type="a(iiay)" access="read"/>
    <property name="AttentionIconName" type="s" access="read"/>
    <property name="AttentionIconPixmap" type="a(iiay)" access="read"/>
    <property name="ToolTip" type="(sa(iiay)ss)" access="read"/>
    <property name="ItemIsMenu" type="b" access="read"/>
    <property name="Menu" type="o" access="read"/>
    <method name="ContextMenu">
      <arg name="x" type="i" direction="in"/>
      <arg name="y" type="i" direction="in"/>
    </method>
    <method name="Activate">
      <arg name="x" type="i" direction="in"/>
      <arg name="y" type="i" direction="in"/>
    </method>
    <method name="SecondaryActivate">
      <arg name="x" type="i" direction="in"/>
      <arg name="y" type="i" direction="in"/>
    </method>
    <method name="Scroll">
      <arg name="delta" type="i" direction="in"/>
      <arg name="orientation" type="s" direction="in"/>
    </method>
    <signal name="NewTitle"/>
    <signal name="NewIcon"/>
    <signal name="NewAttentionIcon"/>
    <signal name="NewOverlayIcon"/>
    <signal name="NewToolTip"/>
    <signal name="NewStatus">
      <arg name="status" type="s"/>
    </signal>
  </interface>
</node>
"""

MENU_XML = f"""
<node>
  <interface name="{MENU_IFACE}">
    <property name="Version" type="u" access="read"/>
    <property name="TextDirection" type="s" access="read"/>
    <property name="Status" type="s" access="read"/>
    <property name="IconThemePath" type="as" access="read"/>
    <method name="GetLayout">
      <arg name="parentId" type="i" direction="in"/>
      <arg name="recursionDepth" type="i" direction="in"/>
      <arg name="propertyNames" type="as" direction="in"/>
      <arg name="revision" type="u" direction="out"/>
      <arg name="layout" type="(ia{{sv}}av)" direction="out"/>
    </method>
    <method name="GetGroupProperties">
      <arg name="ids" type="ai" direction="in"/>
      <arg name="propertyNames" type="as" direction="in"/>
      <arg name="properties" type="a(ia{{sv}})" direction="out"/>
    </method>
    <method name="GetProperty">
      <arg name="id" type="i" direction="in"/>
      <arg name="name" type="s" direction="in"/>
      <arg name="value" type="v" direction="out"/>
    </method>
    <method name="Event">
      <arg name="id" type="i" direction="in"/>
      <arg name="eventId" type="s" direction="in"/>
      <arg name="data" type="v" direction="in"/>
      <arg name="timestamp" type="u" direction="in"/>
    </method>
    <method name="EventGroup">
      <arg name="events" type="a(isvu)" direction="in"/>
      <arg name="idErrors" type="ai" direction="out"/>
    </method>
    <method name="AboutToShow">
      <arg name="id" type="i" direction="in"/>
      <arg name="needUpdate" type="b" direction="out"/>
    </method>
    <method name="AboutToShowGroup">
      <arg name="ids" type="ai" direction="in"/>
      <arg name="updatesNeeded" type="ai" direction="out"/>
      <arg name="idErrors" type="ai" direction="out"/>
    </method>
    <signal name="ItemsPropertiesUpdated">
      <arg name="updatedProps" type="a(ia{{sv}})"/>
      <arg name="removedProps" type="a(ias)"/>
    </signal>
    <signal name="LayoutUpdated">
      <arg name="revision" type="u"/>
      <arg name="parent" type="i"/>
    </signal>
    <signal name="ItemActivationRequested">
      <arg name="id" type="i"/>
      <arg name="timestamp" type="u"/>
    </signal>
  </interface>
</node>
"""


def menu_items(state, detail, service_controls):
    """Build the menu for a state as a list of (id, properties) with plain values."""
    label = STATES[state][1]
    items = [
        (ITEM_STATUS, {'label': f"hyprwhspr: {detail or label}", 'enabled': False}),
        (2, {'type': 'separator'}),
        (ITEM_RECORD, {
            'label': 'Stop recording' if state == 'recording' else 'Start recording',
            'enabled': state in ('ready', 'recording', 'error'),
        }),
        (ITEM_MODEL, {
            'label': 'Load model' if state == 'unloaded' else 'Unload model',
            'enabled': state in ('ready', 'unloaded'),
        }),
    ]
    if service_controls:
        items += [
            (5, {'type': 'separator'}),
            (ITEM_RESTART, {'label': 'Restart hyprwhspr'}),
            (ITEM_STOP, {'label': 'Stop hyprwhspr'}),
        ]
    return items


def menu_action(item_id, state):
    """Map a clicked menu item to an action name, or None."""
    if item_id == ITEM_RECORD:
        return 'stop' if state == 'recording' else 'start'
    if item_id == ITEM_MODEL:
        return 'model_reload' if state == 'unloaded' else 'model_unload'
    if item_id == ITEM_RESTART:
        return 'restart'
    if item_id == ITEM_STOP:
        return 'stop_service'
    return None


def _to_variant(value):
    if isinstance(value, bool):
        return GLib.Variant('b', value)
    return GLib.Variant('s', value)


class TrayIcon:
    """StatusNotifierItem that polls the app for its state and forwards menu clicks"""

    def __init__(self, get_state, on_action, service_controls=False):
        """
        Args:
            get_state: returns (state, detail); state is a key of STATES
            on_action: called with an action name from menu_action()
            service_controls: show Restart/Stop (only meaningful under systemd)
        """
        self.get_state = get_state
        self.on_action = on_action
        self.service_controls = service_controls
        self.bus_name = f"org.kde.StatusNotifierItem-{os.getpid()}-1"
        self._state = 'loading'
        self._detail = ''
        self._revision = 1
        self._context = None
        self._loop = None
        self._thread = None
        self._connection = None
        self._registrations = []
        self._watch_id = 0
        self._ready = threading.Event()
        self._started = False

    def start(self) -> bool:
        """Connect to the session bus and publish the icon in a daemon thread"""
        if not GIO_AVAILABLE:
            print("[TRAY] PyGObject not available, tray icon disabled")
            return False
        if self._thread is not None and self._thread.is_alive():
            return True
        self._ready.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="TrayIcon")
        self._thread.start()
        self._ready.wait(timeout=5.0)
        if not self._started:
            self.stop()
        return self._started

    def stop(self):
        """Remove the icon and stop the thread"""
        loop, context = self._loop, self._context
        if loop is not None and context is not None:
            source = GLib.idle_source_new()
            source.set_callback(lambda *_: loop.quit() or False)
            source.attach(context)
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

    def _run(self):
        """Own a private main context and session bus connection (runs in daemon thread)"""
        self._context = GLib.MainContext.new()
        self._context.push_thread_default()
        try:
            self._connect()
            self._started = True
        except Exception as e:
            print(f"[TRAY] Failed to start: {e}")
            self._started = False
            self._ready.set()
            self._context.pop_thread_default()
            return
        self._ready.set()

        poll = GLib.timeout_source_new(POLL_INTERVAL_MS)
        poll.set_callback(self._poll)
        poll.attach(self._context)
        self._loop = GLib.MainLoop.new(self._context, False)
        try:
            self._loop.run()
        except Exception as e:
            print(f"[TRAY] Main loop crashed: {e}")
        finally:
            poll.destroy()
            self._disconnect()
            self._loop = None
            self._started = False
            self._context.pop_thread_default()

    def _connect(self):
        # A private connection keeps our name and objects separate from other
        # D-Bus users in this process, and closing it removes the icon at once
        address = Gio.dbus_address_get_for_bus_sync(Gio.BusType.SESSION, None)
        self._connection = Gio.DBusConnection.new_for_address_sync(
            address,
            Gio.DBusConnectionFlags.AUTHENTICATION_CLIENT
            | Gio.DBusConnectionFlags.MESSAGE_BUS_CONNECTION,
            None, None)
        self._state, self._detail = self._read_state()

        item_info = Gio.DBusNodeInfo.new_for_xml(ITEM_XML).interfaces[0]
        menu_info = Gio.DBusNodeInfo.new_for_xml(MENU_XML).interfaces[0]
        self._registrations = [
            self._connection.register_object(
                ITEM_PATH, item_info, self._on_item_call, self._get_item_property, None),
            self._connection.register_object(
                MENU_PATH, menu_info, self._on_menu_call, self._get_menu_property, None),
        ]
        self._connection.call_sync(
            'org.freedesktop.DBus', '/org/freedesktop/DBus', 'org.freedesktop.DBus',
            'RequestName', GLib.Variant('(su)', (self.bus_name, 0)),
            None, Gio.DBusCallFlags.NONE, -1, None)
        # Fires now if a tray is running, and again whenever the tray restarts
        self._watch_id = Gio.bus_watch_name_on_connection(
            self._connection, WATCHER_NAME, Gio.BusNameWatcherFlags.NONE,
            self._on_watcher_appeared, self._on_watcher_vanished)

    def _disconnect(self):
        connection = self._connection
        self._connection = None
        if connection is None:
            return
        if self._watch_id:
            Gio.bus_unwatch_name(self._watch_id)
            self._watch_id = 0
        for registration in self._registrations:
            connection.unregister_object(registration)
        self._registrations = []
        try:
            connection.close_sync(None)
        except Exception:
            pass

    def _on_watcher_appeared(self, connection, name, owner):
        connection.call(
            WATCHER_NAME, WATCHER_PATH, WATCHER_NAME, 'RegisterStatusNotifierItem',
            GLib.Variant('(s)', (self.bus_name,)), None, Gio.DBusCallFlags.NONE,
            -1, None, self._on_registered)

    def _on_registered(self, connection, result):
        try:
            connection.call_finish(result)
            print("[TRAY] Tray icon registered")
        except Exception as e:
            print(f"[TRAY] Tray refused the icon: {e}")

    def _on_watcher_vanished(self, connection, name):
        print("[TRAY] No system tray running; the icon appears when one starts")

    def _read_state(self):
        try:
            state, detail = self.get_state()
        except Exception as e:
            return 'error', f"Status unavailable: {e}"
        if state not in STATES:
            return 'error', detail
        return state, detail

    def _poll(self, *_):
        state, detail = self._read_state()
        if (state, detail) != (self._state, self._detail):
            icon_changed = STATES[state][0] != STATES[self._state][0]
            self._state, self._detail = state, detail
            self._revision += 1
            self._emit(ITEM_PATH, ITEM_IFACE, 'NewToolTip', None)
            if icon_changed:
                self._emit(ITEM_PATH, ITEM_IFACE, 'NewIcon', None)
            self._emit(MENU_PATH, MENU_IFACE, 'LayoutUpdated',
                       GLib.Variant('(ui)', (self._revision, 0)))
        return True

    def _emit(self, path, iface, signal, params):
        try:
            self._connection.emit_signal(None, path, iface, signal, params)
        except Exception as e:
            print(f"[TRAY] Failed to emit {signal}: {e}")

    # ==================== StatusNotifierItem ====================

    def _get_item_property(self, connection, sender, path, iface, name):
        icon, label = STATES[self._state]
        values = {
            'Category': GLib.Variant('s', 'ApplicationStatus'),
            'Id': GLib.Variant('s', 'hyprwhspr'),
            'Title': GLib.Variant('s', 'hyprwhspr'),
            'Status': GLib.Variant('s', 'Active'),
            'WindowId': GLib.Variant('i', 0),
            'IconName': GLib.Variant('s', icon),
            'IconThemePath': GLib.Variant('s', ''),
            'IconPixmap': GLib.Variant('a(iiay)', []),
            'OverlayIconName': GLib.Variant('s', ''),
            'OverlayIconPixmap': GLib.Variant('a(iiay)', []),
            'AttentionIconName': GLib.Variant('s', ''),
            'AttentionIconPixmap': GLib.Variant('a(iiay)', []),
            'ToolTip': GLib.Variant('(sa(iiay)ss)', (icon, [], 'hyprwhspr', self._detail or label)),
            'ItemIsMenu': GLib.Variant('b', True),
            'Menu': GLib.Variant('o', MENU_PATH),
        }
        return values.get(name)

    def _on_item_call(self, connection, sender, path, iface, method, params, invocation):
        # ItemIsMenu asks hosts to open the menu on click; nothing to activate
        invocation.return_value(None)

    # ==================== dbusmenu ====================

    def _items(self):
        return menu_items(self._state, self._detail, self.service_controls)

    def _layout(self):
        children = [
            GLib.Variant('(ia{sv}av)', (item_id, {k: _to_variant(v) for k, v in props.items()}, []))
            for item_id, props in self._items()
        ]
        return (0, {'children-display': GLib.Variant('s', 'submenu')}, children)

    def _get_menu_property(self, connection, sender, path, iface, name):
        values = {
            'Version': GLib.Variant('u', 3),
            'TextDirection': GLib.Variant('s', 'ltr'),
            'Status': GLib.Variant('s', 'normal'),
            'IconThemePath': GLib.Variant('as', []),
        }
        return values.get(name)

    def _on_menu_call(self, connection, sender, path, iface, method, params, invocation):
        try:
            args = params.unpack()
            if method == 'GetLayout':
                invocation.return_value(
                    GLib.Variant('(u(ia{sv}av))', (self._revision, self._layout())))
            elif method == 'GetGroupProperties':
                ids = set(args[0])
                props = [
                    (item_id, {k: _to_variant(v) for k, v in item_props.items()})
                    for item_id, item_props in self._items()
                    if not ids or item_id in ids
                ]
                invocation.return_value(GLib.Variant('(a(ia{sv}))', (props,)))
            elif method == 'GetProperty':
                item_props = dict(self._items()).get(args[0], {})
                if args[1] not in item_props:
                    invocation.return_dbus_error(
                        'org.freedesktop.DBus.Error.InvalidArgs', f"No property {args[1]}")
                    return
                invocation.return_value(
                    GLib.Variant('(v)', (_to_variant(item_props[args[1]]),)))
            elif method == 'Event':
                self._handle_event(args[0], args[1])
                invocation.return_value(None)
            elif method == 'EventGroup':
                for item_id, event_id, _data, _timestamp in args[0]:
                    self._handle_event(item_id, event_id)
                invocation.return_value(GLib.Variant('(ai)', ([],)))
            elif method == 'AboutToShow':
                invocation.return_value(GLib.Variant('(b)', (False,)))
            elif method == 'AboutToShowGroup':
                invocation.return_value(GLib.Variant('(aiai)', ([], [])))
            else:
                invocation.return_dbus_error(
                    'org.freedesktop.DBus.Error.UnknownMethod', method)
        except Exception as e:
            print(f"[TRAY] Menu call {method} failed: {e}")
            invocation.return_dbus_error('org.freedesktop.DBus.Error.Failed', str(e))

    def _handle_event(self, item_id, event_id):
        if event_id != 'clicked':
            return
        action = menu_action(item_id, self._state)
        if action is None:
            return
        # Actions can block for seconds (model reload), so keep them off the bus thread
        threading.Thread(target=self.on_action, args=(action,), daemon=True).start()
