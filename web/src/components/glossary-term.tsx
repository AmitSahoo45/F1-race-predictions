import Link from 'next/link';
import type { ReactNode } from 'react';
import { GLOSSARY, type GlossaryId } from '@/lib/glossary';

/** Hover shows the definition; tap or keyboard opens it in the methodology glossary. */
export function GlossaryTerm({ id, children }: { id: GlossaryId; children: ReactNode }) {
  return <Link className="glossary-term" href={`/methodology/#${id}`} title={GLOSSARY[id].definition}>{children}</Link>;
}
