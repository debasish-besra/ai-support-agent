def check_order_status(order_id: str) -> str:
    # Fake database for now
    orders = {
        "12345": "Delivered at 2:30 PM",
        "67890": "Out for delivery",
        "11111": "Cancelled"
    }
    return orders.get(order_id, "Order not found")

def escalate_to_human(reason: str) -> str:
    return f"Escalated to human agent. Reason: {reason}"