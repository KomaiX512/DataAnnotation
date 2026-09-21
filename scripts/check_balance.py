#!/usr/bin/env python3
"""Convenience script to check wallet balance on testnet/finney without password prompts or type errors."""

import argparse
import template.compat.bittensor_commit_hotkey  # noqa: F401 — testnet balance + scale compatibility
import bittensor as bt


def main():
    parser = argparse.ArgumentParser(description="Check wallet balance on Bittensor network")
    parser.add_argument("--wallet.name", "--wallet-name", "--wallet_name", dest="wallet_name", default=None, help="Wallet name")
    parser.add_argument("--wallet.hotkey", "--wallet-hotkey", "--wallet_hotkey", dest="wallet_hotkey", default=None, help="Hotkey name (optional)")
    parser.add_argument("--ss58", dest="ss58", default=None, help="Direct SS58 address to check")
    parser.add_argument("--subtensor.network", "--network", dest="network", default="test", help="Network (test, finney, local)")
    parser.add_argument("--subtensor.chain_endpoint", "--chain_endpoint", dest="chain_endpoint", default=None, help="Chain endpoint")

    args = parser.parse_args()

    target_net = args.chain_endpoint if args.chain_endpoint else args.network
    st = bt.subtensor(network=target_net)

    address = None
    if args.ss58:
        address = args.ss58
    elif args.wallet_name:
        w = bt.wallet(name=args.wallet_name, hotkey=args.wallet_hotkey or "default")
        # Use coldkeypub to avoid password decryption prompts
        address = w.coldkeypub.ss58_address
        print(f"Wallet '{args.wallet_name}' Coldkey SS58: {address}")
        if args.wallet_hotkey:
            print(f"Hotkey '{args.wallet_hotkey}' SS58:  {w.hotkey.ss58_address}")
    else:
        print("Error: Specify either --wallet.name or --ss58")
        return

    bal = st.get_balance(address)
    print(f"Balance on network '{args.network}': {bal}")


if __name__ == "__main__":
    main()
