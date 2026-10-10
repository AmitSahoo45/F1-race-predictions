import { getSiteData } from '@/lib/data';
import { selectCircuitOutline } from '@/lib/circuit-outline';

export const dynamic = 'force-static';

export function generateStaticParams() {
  return getSiteData().events.map((event) => ({ event: event.id }));
}

export async function GET(_request: Request, { params }: { params: Promise<{ event: string }> }) {
  const { event: id } = await params;
  const site = getSiteData();
  const event = site.events.find((item) => item.id === id);
  if (!event) return Response.json(null, { status: 404 });
  return Response.json(selectCircuitOutline(event, site.events, site.analyses) ?? null);
}
