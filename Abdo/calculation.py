from typing import Optional


DEFAULT_VAT_RATE = 0.14
DEFAULT_TOLERANCE = 0.05


# ============================================================
# BASIC CALCULATIONS
# ============================================================

def calculate_net_amount(subtotal: float, discount: float = 0.0) -> float:
    if subtotal < 0:
        raise ValueError("Subtotal cannot be negative.")

    if discount < 0:
        raise ValueError("Discount cannot be negative.")

    if discount > subtotal:
        raise ValueError("Discount cannot exceed subtotal.")

    return subtotal - discount


def calculate_vat(taxable_amount: float, vat_rate: Optional[float] = None) -> float:
    if taxable_amount < 0:
        raise ValueError("Taxable amount cannot be negative.")

    if vat_rate is None:
        vat_rate = DEFAULT_VAT_RATE

    if vat_rate < 0 or vat_rate > 1:
        raise ValueError("VAT rate must be between 0 and 1.")

    return round(taxable_amount * vat_rate, 2)


def calculate_total(net_amount: float, vat_amount: float) -> float:
    if net_amount < 0:
        raise ValueError("Net amount cannot be negative.")

    if vat_amount < 0:
        raise ValueError("VAT amount cannot be negative.")

    return round(net_amount + vat_amount, 2)

# ============================================================
# VALIDATION
# ============================================================

def validate_amount(calculated: float, provided: float, tolerance: float = DEFAULT_TOLERANCE) -> dict:
    """
    Compare a calculated value against the value provided
    on the invoice.
    """

    if tolerance < 0:
        raise ValueError("Tolerance cannot be negative.")

    difference = calculated - provided

    return {
        "valid": abs(difference) <= tolerance,
        "calculated": calculated,
        "provided": provided,
        "difference": difference
    }


# ============================================================
# COMPLETE INVOICE CALCULATION
# ============================================================

def calculate_invoice(subtotal: float, discount: float = 0.0, vat_rate: Optional[float] = None) -> dict:
    """
    Perform all invoice calculations.

    Flow:

        Subtotal
            ↓
        Discount
            ↓
        Net Amount
            ↓
        VAT
            ↓
        Total
    """

    if vat_rate is None:
        vat_rate = DEFAULT_VAT_RATE

    net_amount = calculate_net_amount(subtotal=subtotal, discount=discount)

    vat_amount = calculate_vat(taxable_amount=net_amount, vat_rate=vat_rate)

    total = calculate_total(net_amount=net_amount, vat_amount=vat_amount)

    return {
        "subtotal": subtotal,
        "discount": discount,
        "net_amount": net_amount,
        "vat_rate": vat_rate,
        "vat_amount": vat_amount,
        "total": total
    }


# ============================================================
# COMPLETE INVOICE VALIDATION
# ============================================================

def validate_invoice(
    subtotal: float,
    discount: float,
    vat_rate: float,
    provided_vat: float,
    provided_total: float,
    tolerance: float = DEFAULT_TOLERANCE
) -> dict:
    """
    Calculate the expected invoice values and compare them
    against the values extracted from the invoice.
    """

    calculated = calculate_invoice(
        subtotal=subtotal,
        discount=discount,
        vat_rate=vat_rate
    )

    vat_validation = validate_amount(
        calculated=calculated["vat_amount"],
        provided=provided_vat,
        tolerance=tolerance
    )

    total_validation = validate_amount(
        calculated=calculated["total"],
        provided=provided_total,
        tolerance=tolerance
    )

    return {
        "calculation": calculated,
        "vat_validation": vat_validation,
        "total_validation": total_validation,
        "invoice_valid": (
            vat_validation["valid"]
            and total_validation["valid"]
        )
    }