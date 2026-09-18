import os
from datetime import datetime

import bittensor


def main(args):
    wallet = bittensor.Wallet(name=args.name)
    password = getattr(args, "password", None) or os.environ.get("WALLET_PASS")
    if password:
        keypair = wallet.get_coldkey(password=password)
    else:
        keypair = wallet.coldkey

    timestamp = datetime.now()
    timezone = timestamp.astimezone().tzname()

    # ensure compatiblity with polkadotjs messages, as polkadotjs always wraps message
    message = (
        "<Bytes>" + f"On {timestamp} {timezone} {args.message}" + "</Bytes>"
    )
    signature = keypair.sign(data=message)

    file_contents = f"{message}\n\tSigned by: {keypair.ss58_address}\n\tSignature: {signature.hex()}"
    print(file_contents)
    open("message_and_signature.txt", "w").write(file_contents)

    print("Signature generated and saved to message_and_signature.txt")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate a signature")
    parser.add_argument("--message", help="The message to sign", type=str)
    parser.add_argument("--name", help="The wallet name", type=str)
    parser.add_argument("--password", help="The wallet password", type=str, default=None)
    args = parser.parse_args()

    main(args)

