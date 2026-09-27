import type { Metadata } from 'next';
import localFont from 'next/font/local';
import { getSiteData } from '@/lib/data';
import { DemoBanner, SiteFooter, SiteHeader } from '@/components/site-shell';
import './globals.css';

const space = localFont({ src: '../../node_modules/@fontsource-variable/space-grotesk/files/space-grotesk-latin-wght-normal.woff2', variable: '--font-space', display: 'optional' });
const plex = localFont({ src: '../../node_modules/@fontsource/ibm-plex-mono/files/ibm-plex-mono-latin-400-normal.woff2', variable: '--font-plex', display: 'optional' });


export const metadata: Metadata = {
  metadataBase: new URL(process.env.NEXT_PUBLIC_SITE_URL ?? (process.env.GITHUB_REPOSITORY_OWNER ? `https://${process.env.GITHUB_REPOSITORY_OWNER.toLowerCase()}.github.io` : 'http://localhost:3100')),
  title: { default: 'APEX FORECAST — Before the lights go out.', template: '%s | APEX FORECAST' },
  description: 'An independent Formula 1 prediction project. Explore qualifying and race forecasts, public data, and honest model evidence.',
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  const { demo_notice, events } = getSiteData();
  const season = events.length ? Math.max(...events.map((event) => event.season)) : new Date().getUTCFullYear();
  return <html lang="en" data-theme="dark" className={`${space.variable} ${plex.variable}`}>
    <body><a className="skip-link" href="#main">Skip to content</a><SiteHeader season={season} /><DemoBanner notice={demo_notice} />{children}<SiteFooter /></body>
  </html>;
}
