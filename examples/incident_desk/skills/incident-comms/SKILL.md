---
name: incident-comms
description: Word and publish the incident status update once a ticket exists and on-call has been paged.
allowed-tools: post_status_update
---
# Incident communications

- Call `post_status_update` only after a ticket has been created and on-call paged in this
  conversation; before that, restate what is pending and post nothing.
- Follow references/status-update-template.md: name the service, the severity and the
  ticket id in the update.
- Mention the ticket id in the final message. If the update fails, say the status page
  could not be reached.
