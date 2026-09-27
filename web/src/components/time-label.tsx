'use client';

import { useEffect, useState } from 'react';

export function TimeLabel({ iso, trackZone, detail = false }: { iso: string; trackZone?: string; detail?: boolean }) {
  const [viewer, setViewer] = useState<string>();
  useEffect(() => {
    setViewer(new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' }).format(new Date(iso)));
  }, [iso]);
  const utc = new Intl.DateTimeFormat('en-GB', { timeZone: 'UTC', dateStyle: 'medium', timeStyle: 'short' }).format(new Date(iso));
  const track = trackZone ? new Intl.DateTimeFormat('en-GB', { timeZone: trackZone, dateStyle: 'medium', timeStyle: 'short' }).format(new Date(iso)) : null;
  return <time dateTime={iso} suppressHydrationWarning>{viewer ?? `${utc} UTC`}{detail && track ? <small className="track-time">Track local: {track}</small> : null}</time>;
}
