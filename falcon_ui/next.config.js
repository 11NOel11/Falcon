/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  output: 'export',
  images: {
    unoptimized: true,
  },
  // Set basePath for GitHub Pages (repo name)
  // Comment this out for local development
  basePath: process.env.NODE_ENV === 'production' ? '/Falcon' : '',
  assetPrefix: process.env.NODE_ENV === 'production' ? '/Falcon/' : '',
}

module.exports = nextConfig
