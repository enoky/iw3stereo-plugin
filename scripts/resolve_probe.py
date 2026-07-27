"""Ask a running DaVinci Resolve what it is and what OFX plugins it registered.

Answers Phase 0d (edition, and did our plugin load) without touching the UI.
Resolve must already be running, with a project open, and external scripting
enabled at Preferences -> System -> General -> "External scripting using: Local".

    F:\\_AI_PROJECTS_\\nunif\\venv\\Scripts\\python.exe scripts\\resolve_probe.py

Add --fusion to also enumerate the inputs of the probe plugins as Fusion tools,
which is the headless way to answer Phase 0c (does a second input clip exist).
That one creates a scratch Fusion composition in the current project.
"""

import argparse
import os
import sys

RESOLVE_SCRIPT_API = os.path.join(
    os.environ.get("PROGRAMDATA", r"C:\ProgramData"),
    r"Blackmagic Design\DaVinci Resolve\Support\Developer\Scripting")
RESOLVE_SCRIPT_LIB = r"C:\Program Files\Blackmagic Design\DaVinci Resolve\fusionscript.dll"

PROBE_TOOL_IDS = [
    "ofx.com.nunif.iw3.probe.single",
    "ofx.com.nunif.iw3.probe.dual",
]


def connect():
    os.environ.setdefault("RESOLVE_SCRIPT_API", RESOLVE_SCRIPT_API)
    os.environ.setdefault("RESOLVE_SCRIPT_LIB", RESOLVE_SCRIPT_LIB)
    sys.path.append(os.path.join(RESOLVE_SCRIPT_API, "Modules"))

    try:
        import DaVinciResolveScript as bmd
    except ImportError:
        sys.exit(f"could not import DaVinciResolveScript from {RESOLVE_SCRIPT_API}\\Modules")

    resolve = bmd.scriptapp("Resolve")
    if resolve is None:
        sys.exit("Resolve is not running, or external scripting is disabled "
                 "(Preferences -> System -> General -> External scripting using: Local)")
    return resolve


def report_edition(resolve):
    print("== edition ==")
    # This is the authoritative check: free Resolve reports "DaVinci Resolve",
    # Studio reports "DaVinci Resolve Studio".
    print(f"  product : {resolve.GetProductName()}")
    print(f"  version : {resolve.GetVersionString()}")
    print(f"  page    : {resolve.GetCurrentPage()}")


def report_project(resolve):
    print("== project ==")
    project = resolve.GetProjectManager().GetCurrentProject()
    if project is None:
        print("  no project open")
        return None
    print(f"  name     : {project.GetName()}")
    timeline = project.GetCurrentTimeline()
    print(f"  timeline : {timeline.GetName() if timeline else '<none>'}")
    for key in ("colorScienceMode", "colorSpaceTimeline", "colorSpaceInput",
                "colorSpaceOutput", "timelineResolutionWidth", "timelineResolutionHeight"):
        print(f"  {key:24s}: {project.GetSetting(key)}")
    return project


def report_fusion_tools(resolve):
    """Add each probe plugin as a Fusion tool and list the inputs it exposes.

    An OFX plugin appears in Fusion as a tool named ofx.<plugin identifier>.
    If AddTool returns something, Resolve registered the plugin. The tool's
    input list then shows whether the second clip made it through.
    """
    print("== fusion ==")
    fusion = resolve.Fusion()
    if fusion is None:
        print("  no Fusion object")
        return

    comp = fusion.NewComp()
    if comp is None:
        print("  could not create a scratch composition; open the Fusion page and retry")
        return

    try:
        for tool_id in PROBE_TOOL_IDS:
            tool = comp.AddTool(tool_id)
            if tool is None:
                print(f"  {tool_id}: NOT REGISTERED")
                continue
            print(f"  {tool_id}: registered")
            inputs = tool.GetInputList() or {}
            for index in sorted(inputs.keys()):
                item = inputs[index]
                try:
                    attrs = item.GetAttrs()
                    name = attrs.get("INPS_Name", "?")
                    kind = attrs.get("INPS_ID", "?")
                    print(f"      input {index}: {name}  ({kind})")
                except Exception as error:  # noqa: BLE001 - report and continue
                    print(f"      input {index}: <unreadable: {error}>")
    finally:
        comp.Close()


def report_probe_log():
    print("== probe log ==")
    path = os.path.join(os.environ.get("LOCALAPPDATA", ""), "iw3probe", "probe.log")
    if not os.path.exists(path):
        print(f"  {path} does not exist -- Resolve never loaded the probe plugin")
        return
    print(f"  {path}")
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            print("  | " + line.rstrip())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fusion", action="store_true",
                        help="add the probe plugins as Fusion tools and list their inputs")
    args = parser.parse_args()

    resolve = connect()
    report_edition(resolve)
    report_project(resolve)
    if args.fusion:
        report_fusion_tools(resolve)
    report_probe_log()


if __name__ == "__main__":
    main()
