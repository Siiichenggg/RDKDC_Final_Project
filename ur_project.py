from __future__ import annotations

import sys

from IK_control import run_ik_mode
from rr_control import DEFAULT_HOME_Q, return_home, run_rr_mode
from ur_interface import UrInterface


def main() -> None:
    """Simple CLI menu to select the control strategy."""

    ur = UrInterface()
    home_q = DEFAULT_HOME_Q.copy()
    print("Using configured home configuration.")

    while True:
        print("\nSelect control mode:")
        print("[1] Resolved-Rate (RR) control")
        print("[2] IK control")
        print("[0] Exit")
        choice = input("Enter choice: ").strip()

        if choice == "1":
            run_rr_mode(ur, home_q)
        elif choice == "2":
            run_ik_mode(ur, home_q)
        elif choice == "0":
            try:
                return_home(ur, home_q)
            except Exception as exc:
                print(f"Failed to return home before exit: {exc}")
            break
        else:
            print("Invalid choice.")

    print("Exiting...")
    sys.exit(0)


if __name__ == "__main__":
    main()
