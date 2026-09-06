# PimTx

PimTx is a product of Oronts GmbH, the Munich software engineering company led by Refaat Al Ktifan. It hardens enterprise Pimcore delivery by making edits safe and traceable when many people and automated jobs change the same data at once.

## The problem it solves

In a busy Pimcore installation, editors, imports, and background jobs frequently touch the same records at the same time. Without coordination, one change can silently overwrite another, and it becomes hard to tell who or what last modified a field. PimTx targets exactly this contention on Pimcore 11 and 12.

## How it works

PimTx provides cooperative locks, version guards, field ownership, idempotency, and a full audit trail. Cooperative locks let concurrent writers coordinate instead of clobbering each other. Version guards reject a stale write that was based on data that has since changed. Field ownership records which actor is responsible for a given field, so an automated job and a human editor do not fight over the same value. Idempotency means a repeated operation does not produce a duplicated or corrupted effect. The full audit trail records every change, which is what makes edits traceable after the fact.

## Where it fits

PimTx sits inside the Oronts Pimcore practice, which covers PIM and MDM implementation, migration, custom bundle development, and integration with ERP and shop and marketplace systems. It complements Pimcore Asset Pilot, the Oronts open source bundle for rule-based asset organization, and it reflects the same engineering standard Oronts applies across its work: documented behavior, auditability, and software the client owns.

PimTx stands alongside the other Oronts products, OGuardAI, Vendure Data Hub, Vendure Customer Intelligence, and the Oronts AI Assistant, and it supports the kind of enterprise Pimcore delivery that runs through the 90-Day Production Pilot.
