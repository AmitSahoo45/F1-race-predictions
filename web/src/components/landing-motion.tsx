'use client';

import { useEffect } from 'react';

export function LandingMotion() {
  useEffect(() => {
    let active = true;
    let context: { revert: () => void } | undefined;
    const begin = async () => {
      if (!active) return;
      const [{ default: gsap }, { ScrollTrigger }] = await Promise.all([import('gsap'), import('gsap/ScrollTrigger')]);
      if (!active) return;
      gsap.registerPlugin(ScrollTrigger);
      context = gsap.context(() => {
        const media = gsap.matchMedia();
        media.add('(prefers-reduced-motion: no-preference)', () => {
          gsap.fromTo('.scroll-reveal', { y: 28, opacity: 0 }, { y: 0, opacity: 1, duration: 0.7, stagger: 0.1, scrollTrigger: { trigger: '.scroll-reveal', start: 'top 88%' } });
        });
        return () => media.revert();
      }, document.querySelector('main#main') ?? document.body);
    };
    window.addEventListener('scroll', begin, { once: true, passive: true });
    return () => { active = false; window.removeEventListener('scroll', begin); context?.revert(); };
  }, []);
  return null;
}
