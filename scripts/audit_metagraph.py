#!/usr/bin/env python3
"""
Metagraph & Incentive Audit Script
Subnet 498 (Testnet) - Climate MRV Subnet
"""

import os
import sys
import json
import time
import asyncio
from pathlib import Path

import template.compat.bittensor_commit_hotkey
import bittensor as bt
from dotenv import load_dotenv

load_dotenv()

def print_header(title):
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)

def audit_metagraph():
    print_header("1. METAGRAPH ON-CHAIN STATUS (NETUID 498)")
    sub = bt.subtensor(network="test")
    mg = sub.metagraph(netuid=498)
    
    print(f"Subnet Netuid: {mg.netuid} | Total Neurons: {mg.n}")
    print(f"{'UID':<5} {'Hotkey SS58':<48} {'IP:Port':<22} {'Stake (TAO)':<12} {'Incentive':<10} {'Emission':<10} {'Permit':<8}")
    print("-" * 120)
    
    nodes = []
    for i in range(mg.n):
        hotkey = mg.hotkeys[i]
        axon = mg.axons[i]
        stake = float(mg.S[i])
        inc = float(mg.I[i])
        emm = float(mg.E[i])
        permit = bool(mg.validator_permit[i])
        ip_str = f"{axon.ip}:{axon.port}"
        
        print(f"{i:<5} {hotkey:<48} {ip_str:<22} {stake:<12.4f} {inc:<10.4f} {emm:<10.4f} {str(permit):<8}")
        nodes.append({
            "uid": i,
            "hotkey": hotkey,
            "ip_port": ip_str,
            "stake": stake,
            "incentive": inc,
            "emission": emm,
            "validator_permit": permit,
        })
    return nodes

if __name__ == "__main__":
    audit_metagraph()
