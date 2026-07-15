#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Minicoder Vehicle AI - Demo Script
No API Key Required - Shows All Core Features
"""

import sys
import subprocess
import json

# Test cases
test_cases = [
    {
        "name": "Demo 1: Simple Range Query",
        "input": "查询A102的续航",
        "desc": "Show basic intent recognition and entity extraction"
    },
    {
        "name": "Demo 2: Missing Slot Detection",
        "input": "明天去上海，需要充电吗",
        "desc": "Show missing field detection (vehicle_id, departure_time)"
    },
    {
        "name": "Demo 3: Complete Trip Planning",
        "input": "A102明早8点去上海虹桥站，判断是否需要充电",
        "desc": "Show 6-step task planning (status->route->energy->station->advice->knowledge)"
    },
    {
        "name": "Demo 4: Knowledge Query",
        "input": "冬天如何保养电池",
        "desc": "Show knowledge retrieval intent recognition"
    },
    {
        "name": "Demo 5: Climate Control",
        "input": "把A102的空调设为24度",
        "desc": "Show control intent and temperature entity extraction"
    }
]

def print_separator():
    print("\n" + "=" * 80 + "\n")

def run_demo(test_case):
    """Run a single demo"""
    print(f"[{test_case['name']}]")
    print(f"Input: {test_case['input']}")
    print(f"Description: {test_case['desc']}")
    print("-" * 80)

    # Run command
    cmd = [
        sys.executable, "-c",
        f"from minicoder.cli import main; import sys; "
        f"sys.argv = ['minicoder', '--analyze-intent', '{test_case['input']}']; "
        f"main()"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10, encoding='utf-8', errors='replace')

        # Parse JSON output
        output = result.stdout.strip()
        if not output:
            print("No output received")
            if result.stderr:
                print(f"Error: {result.stderr}")
            return

        try:
            data = json.loads(output)

            # Extract key information
            understanding = data.get('understanding', {})
            plan = data.get('plan', {})

            print(f"\n* Intent: {understanding.get('intent', 'N/A')}")
            print(f"* Status: {understanding.get('status', 'N/A')}")
            print(f"* Confidence: {understanding.get('confidence', 'N/A')}")

            entities = understanding.get('entities', {})
            if entities:
                print(f"* Extracted Entities: {json.dumps(entities, ensure_ascii=False)}")

            missing = understanding.get('missing_fields', [])
            if missing:
                print(f"* Missing Fields: {', '.join(missing)}")

            steps = plan.get('steps', [])
            if steps:
                print(f"\n* Generated Plan: {len(steps)} steps")
                for i, step in enumerate(steps, 1):
                    deps = step.get('depends_on', [])
                    dep_str = f" (depends: {', '.join(deps)})" if deps else ""
                    print(f"  {i}. {step['action']}{dep_str}")

        except json.JSONDecodeError as e:
            print(f"JSON parse error at position {e.pos}")
            print("\nRaw output:")
            print(output[:500])  # Show first 500 chars
            if result.stderr:
                print(f"\nStderr: {result.stderr}")

    except subprocess.TimeoutExpired:
        print("WARNING: Execution timeout")
    except Exception as e:
        print(f"WARNING: Execution error: {e}")

def main():
    print_separator()
    print("Vehicle AI Intelligent Assistant - Project Demo")
    print("Core Features Demo (No API Key Required)")
    print_separator()

    for i, test_case in enumerate(test_cases):
        run_demo(test_case)

        if i < len(test_cases) - 1:
            print("\n" + "-" * 80)
            input("\nPress Enter to continue to next demo...")
            print()

    print_separator()
    print("Demo Complete!")
    print("\nCore Capabilities:")
    print("  * Intent Recognition (7 vehicle intents)")
    print("  * Entity Extraction (vehicle_id, location, time, temperature)")
    print("  * Slot Management (missing field detection)")
    print("  * Task Planning (auto-generated execution steps)")
    print("\nMock Vehicle API: http://127.0.0.1:8765/docs")
    print_separator()

    input("Press Enter to exit...")

if __name__ == "__main__":
    main()
