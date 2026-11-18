# Creates the launcher shortcut next to this script. Run it again if the project folder moves:
#   powershell -NoProfile -ExecutionPolicy Bypass -File make-shortcut.ps1
# The shortcut runs start-gui.cmd (minimized console), uses assets\taffy-avatar.ico (or assets\app-icon.ico), and carries the same
# AppUserModelID that gui\taskbar.py sets, so a taskbar pin of this shortcut and the running window share one icon.
# Keep this file ASCII: Windows PowerShell 5.1 reads scripts without a BOM in the ANSI code page.
param(
    [string]$Name = ([string][char]0x89D2 + [char]0x8272 + [char]0x53D8 + [char]0x58F0 + [char]0x5668),
    [string]$AppId = 'LocalVoiceChanger.Gui',
    [switch]$RepairPinned
)
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot

Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices.ComTypes;
using System.Text;

namespace VoiceGui {
    [ComImport, Guid("00021401-0000-0000-C000-000000000046")]
    class CShellLink { }

    [ComImport, InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("000214F9-0000-0000-C000-000000000046")]
    interface IShellLinkW {
        void GetPath([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder file, int cch, IntPtr findData, uint flags);
        void GetIDList(out IntPtr pidl);
        void SetIDList(IntPtr pidl);
        void GetDescription([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder name, int cch);
        void SetDescription([MarshalAs(UnmanagedType.LPWStr)] string name);
        void GetWorkingDirectory([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder dir, int cch);
        void SetWorkingDirectory([MarshalAs(UnmanagedType.LPWStr)] string dir);
        void GetArguments([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder args, int cch);
        void SetArguments([MarshalAs(UnmanagedType.LPWStr)] string args);
        void GetHotkey(out short hotkey);
        void SetHotkey(short hotkey);
        void GetShowCmd(out int showCmd);
        void SetShowCmd(int showCmd);
        void GetIconLocation([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder path, int cch, out int index);
        void SetIconLocation([MarshalAs(UnmanagedType.LPWStr)] string path, int index);
        void SetRelativePath([MarshalAs(UnmanagedType.LPWStr)] string path, uint reserved);
        void Resolve(IntPtr hwnd, uint flags);
        void SetPath([MarshalAs(UnmanagedType.LPWStr)] string file);
    }

    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    struct PROPERTYKEY { public Guid fmtid; public uint pid; }

    [StructLayout(LayoutKind.Explicit, Size = 24)]
    struct PROPVARIANT {
        [FieldOffset(0)] public ushort vt;
        [FieldOffset(8)] public IntPtr pointer;
    }

    [ComImport, InterfaceType(ComInterfaceType.InterfaceIsIUnknown), Guid("886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99")]
    interface IPropertyStore {
        void GetCount(out uint count);
        void GetAt(uint index, out PROPERTYKEY key);
        void GetValue(ref PROPERTYKEY key, out PROPVARIANT value);
        void SetValue(ref PROPERTYKEY key, ref PROPVARIANT value);
        void Commit();
    }

    public static class Shortcut {
        static PROPERTYKEY AppIdKey = new PROPERTYKEY { fmtid = new Guid("9F4C2855-9F79-4B39-A8D0-E1D42DE1D5F3"), pid = 5 };

        public static void Create(string lnk, string target, string args, string workDir, string icon,
                                  string description, string appId) {
            IShellLinkW link = (IShellLinkW)new CShellLink();
            link.SetPath(target);
            link.SetArguments(args);
            link.SetWorkingDirectory(workDir);
            link.SetIconLocation(icon, 0);
            link.SetDescription(description);
            link.SetShowCmd(7);  // minimized, not activated
            IPropertyStore store = (IPropertyStore)link;
            PROPVARIANT value = new PROPVARIANT { vt = 31, pointer = Marshal.StringToCoTaskMemUni(appId) };  // VT_LPWSTR
            try {
                PROPERTYKEY key = AppIdKey;
                store.SetValue(ref key, ref value);
                store.Commit();
            } finally {
                Marshal.FreeCoTaskMem(value.pointer);
            }
            ((IPersistFile)link).Save(lnk, true);
        }

        public static string ReadAppId(string lnk) {
            IShellLinkW link = (IShellLinkW)new CShellLink();
            ((IPersistFile)link).Load(lnk, 0);
            PROPVARIANT value;
            PROPERTYKEY key = AppIdKey;
            ((IPropertyStore)link).GetValue(ref key, out value);
            string text = value.vt == 31 ? Marshal.PtrToStringUni(value.pointer) : null;
            if (value.vt == 31) Marshal.FreeCoTaskMem(value.pointer);
            return text;
        }
    }
}
"@

$lnk = Join-Path $root ($Name + '.lnk')
$cmd = Join-Path $env:SystemRoot 'System32\cmd.exe'
$icon = Join-Path $root 'assets\taffy-avatar.ico'
if (-not (Test-Path -LiteralPath $icon)) { $icon = Join-Path $root 'assets\app-icon.ico' }  # public edition has no character art
if (-not (Test-Path -LiteralPath (Join-Path $root 'start-gui.cmd'))) { throw "start-gui.cmd not found in $root" }
if (-not (Test-Path -LiteralPath $icon)) { throw "icon not found: $icon" }
[VoiceGui.Shortcut]::Create($lnk, $cmd, '/d /c ".\start-gui.cmd"', $root, $icon, $Name, $AppId)
"created: $lnk"
"app id:  $([VoiceGui.Shortcut]::ReadAppId($lnk))"

# Moving the project does not update Windows' separate copy of a taskbar pin.
# Repair an existing pin only; this never pins the app or changes unrelated pins.
if ($RepairPinned) {
    $pinned = Join-Path $env:APPDATA ('Microsoft\Internet Explorer\Quick Launch\User Pinned\TaskBar\' + $Name + '.lnk')
    if (Test-Path -LiteralPath $pinned) {
        if ([VoiceGui.Shortcut]::ReadAppId($pinned) -ne $AppId) {
            throw 'The existing taskbar shortcut has a different app ID; leaving it unchanged.'
        }
        $backup = Join-Path $root ('.run\shortcut-backup-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
        New-Item -ItemType Directory -Path $backup -Force | Out-Null
        Copy-Item -LiteralPath $pinned -Destination $backup
        Copy-Item -LiteralPath $lnk -Destination $pinned -Force
        Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class VoiceShortcutNotify {
    [DllImport("shell32.dll", CharSet = CharSet.Unicode)]
    public static extern void SHChangeNotify(uint change, uint flags, string item, IntPtr other);
}
"@
        [VoiceShortcutNotify]::SHChangeNotify(0x2000, 0x1005, $pinned, [IntPtr]::Zero)
        [VoiceShortcutNotify]::SHChangeNotify(0x08000000, 0, $null, [IntPtr]::Zero)
        "repaired existing taskbar shortcut: $pinned"
    }
}
