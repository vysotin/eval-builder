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
