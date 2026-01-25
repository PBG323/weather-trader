"""
Polymarket Authentication

Handles wallet management and API authentication for Polymarket CLOB.

Polymarket uses:
- Polygon network for trading
- USDC as the trading currency
- Non-custodial wallet-based authentication
"""

import os
from dataclasses import dataclass
from typing import Optional
import secrets

from eth_account import Account
from web3 import Web3

from ..config import config


@dataclass
class WalletInfo:
    """Information about a trading wallet."""
    address: str
    private_key: str
    chain_id: int = 137  # Polygon mainnet


def create_wallet() -> WalletInfo:
    """
    Create a new Ethereum/Polygon wallet for trading.

    WARNING: Store the private key securely! Loss of private key
    means loss of all funds.

    Returns:
        WalletInfo with address and private key
    """
    # Generate random private key
    private_key = "0x" + secrets.token_hex(32)
    account = Account.from_key(private_key)

    return WalletInfo(
        address=account.address,
        private_key=private_key,
        chain_id=137,
    )


def load_wallet_from_env() -> Optional[WalletInfo]:
    """
    Load wallet from environment variables.

    Returns:
        WalletInfo if private key is configured, None otherwise
    """
    private_key = config.polygon.private_key

    if not private_key:
        return None

    # Ensure proper format
    if not private_key.startswith("0x"):
        private_key = "0x" + private_key

    try:
        account = Account.from_key(private_key)
        return WalletInfo(
            address=account.address,
            private_key=private_key,
            chain_id=config.polygon.chain_id,
        )
    except Exception as e:
        raise ValueError(f"Invalid private key in configuration: {e}")


class PolymarketAuth:
    """
    Authentication manager for Polymarket.

    Handles:
    - Wallet loading and validation
    - Message signing for API authentication
    - Balance checking
    """

    def __init__(self, wallet: Optional[WalletInfo] = None):
        self.wallet = wallet or load_wallet_from_env()
        self.w3 = Web3(Web3.HTTPProvider(config.polygon.rpc_url))

        if self.wallet:
            self.account = Account.from_key(self.wallet.private_key)
        else:
            self.account = None

    @property
    def is_configured(self) -> bool:
        """Check if authentication is properly configured."""
        return self.wallet is not None and self.account is not None

    @property
    def address(self) -> Optional[str]:
        """Get the wallet address."""
        return self.wallet.address if self.wallet else None

    def sign_message(self, message: str) -> str:
        """
        Sign a message for API authentication.

        Args:
            message: Message to sign

        Returns:
            Hex-encoded signature
        """
        if not self.account:
            raise ValueError("No wallet configured for signing")

        message_hash = Web3.keccak(text=message)
        signed = self.account.signHash(message_hash)
        return signed.signature.hex()

    def get_usdc_balance(self) -> float:
        """
        Get USDC balance on Polygon.

        Returns:
            USDC balance (human-readable, 6 decimals adjusted)
        """
        if not self.wallet:
            return 0.0

        # USDC contract on Polygon
        usdc_address = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"

        # Minimal ERC20 ABI for balanceOf
        erc20_abi = [
            {
                "constant": True,
                "inputs": [{"name": "_owner", "type": "address"}],
                "name": "balanceOf",
                "outputs": [{"name": "balance", "type": "uint256"}],
                "type": "function",
            }
        ]

        try:
            contract = self.w3.eth.contract(
                address=Web3.to_checksum_address(usdc_address),
                abi=erc20_abi
            )
            balance_wei = contract.functions.balanceOf(
                Web3.to_checksum_address(self.wallet.address)
            ).call()

            # USDC has 6 decimals
            return balance_wei / 1_000_000

        except Exception as e:
            print(f"Error fetching USDC balance: {e}")
            return 0.0

    def get_matic_balance(self) -> float:
        """
        Get MATIC balance for gas fees.

        Returns:
            MATIC balance in Ether units
        """
        if not self.wallet:
            return 0.0

        try:
            balance_wei = self.w3.eth.get_balance(
                Web3.to_checksum_address(self.wallet.address)
            )
            return self.w3.from_wei(balance_wei, "ether")

        except Exception as e:
            print(f"Error fetching MATIC balance: {e}")
            return 0.0

    def get_balances(self) -> dict:
        """
        Get all relevant balances.

        Returns:
            Dictionary with USDC and MATIC balances
        """
        return {
            "usdc": self.get_usdc_balance(),
            "matic": self.get_matic_balance(),
            "address": self.address,
        }

    def validate_for_trading(self, min_usdc: float = 10.0, min_matic: float = 0.1) -> tuple[bool, str]:
        """
        Validate that wallet is ready for trading.

        Args:
            min_usdc: Minimum USDC balance required
            min_matic: Minimum MATIC balance for gas

        Returns:
            Tuple of (is_valid, reason)
        """
        if not self.is_configured:
            return False, "Wallet not configured. Set POLYGON_PRIVATE_KEY in .env"

        usdc = self.get_usdc_balance()
        if usdc < min_usdc:
            return False, f"Insufficient USDC: {usdc:.2f} < {min_usdc:.2f}"

        matic = self.get_matic_balance()
        if matic < min_matic:
            return False, f"Insufficient MATIC for gas: {matic:.4f} < {min_matic:.4f}"

        return True, "Wallet ready for trading"
