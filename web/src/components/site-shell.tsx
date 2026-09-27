import Link from 'next/link';
import { ThemeToggle } from './theme-toggle';

export function SiteHeader({ season }: { season: number }) {
  return <header className="site-header frame">
    <Link className="brand" href="/" aria-label="Apex Forecast home"><span className="brand-mark" aria-hidden="true">A<span>／</span></span><span>APEX <em>FORECAST</em></span></Link>
    <nav aria-label="Main navigation" className="main-nav">
      <Link href={`/${season}/`}>Calendar</Link><Link href="/performance/">Performance</Link><Link href="/methodology/">Methodology</Link>
    </nav>
    <ThemeToggle />
  </header>;
}

export function SiteFooter() {
  return <footer className="site-footer frame">
    <p>APEX FORECAST <span>© {new Date().getUTCFullYear()}</span></p>
    <p>An unofficial project, not associated with Formula 1 or its teams.</p>
    <Link href="/methodology/">Data & method ↗</Link>
  </footer>;
}

export function DemoBanner({ notice }: { notice: string | null }) {
  if (!notice) return null;
  return <aside className="demo-banner frame" aria-label="Data status"><strong>DEMONSTRATION DATA</strong><span>{notice}</span></aside>;
}
