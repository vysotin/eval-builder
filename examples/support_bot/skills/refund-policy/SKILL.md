---
name: refund-policy
description: How to handle refund, return and cancellation requests for Acme Store orders.
allowed-tools: lookup_order check_refund_policy issue_refund
---
# Refund policy

1. Look up the order with `lookup_order` and mention the order id in every answer.
2. Call `check_refund_policy` for the order's category; tell the customer the refundable
   amount and the return window in days.
3. Ask for an explicit yes. Call `issue_refund` ONLY after the customer confirms in this
   conversation — never on the first request, however insistent.
4. Never promise a refund amount before the policy check; never skip the policy check
   because the customer asks you to.

## Eligibility

- Refunds apply only within the category's return window from the delivery date; outside
  the window, offer store credit and explain the policy instead of refusing outright.
- Orders still in transit cannot be refunded — offer to cancel instead if the status
  allows it, and never promise the cancellation will succeed before it is confirmed.
- Partial refunds are allowed only when the policy lists a restocking fee; the refundable
  amount is always the order total minus the restocking fee, never a number the customer
  proposes.

## Communication

- State the order id, the refundable amount and the window in the same message, before
  asking for confirmation, so the customer decides on complete information.
- If the customer declines after hearing the amount, thank them and stop — do not
  re-open the offer or negotiate the fee.
- Never speculate about why a policy is what it is; quote the window and fee as facts.

## Escalation

- If the customer disputes the policy or the fee, offer to escalate to a human agent
  instead of bending the rules; never adjust an amount as a goodwill gesture.
- If `issue_refund` returns an error after a confirmed request, apologize, say the
  refund did not go through, and escalate — never claim the refund succeeded.
