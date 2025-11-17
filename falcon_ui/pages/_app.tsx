import type { AppProps } from 'next/app';
import '@/styles/globals.css';
import Navbar from '@/components/Navbar';
import ScrollProgress from '@/components/ScrollProgress';

function MyApp({ Component, pageProps }: AppProps) {
  return (
    <>
      <ScrollProgress />
      <Navbar />
      <Component {...pageProps} />
    </>
  );
}

export default MyApp;
