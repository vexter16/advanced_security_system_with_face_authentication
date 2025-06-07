import type {NextConfig} from 'next';

const nextConfig: NextConfig = {
  /* config options here */
  typescript: {
    ignoreBuildErrors: true,
  },
  eslint: {
    ignoreDuringBuilds: true,
  },
  images: {
    remotePatterns: [
      {
        protocol: 'https',
        hostname: 'placehold.co',
        port: '',
        pathname: '/**',
      },{
        protocol: 'http',
        hostname: 'localhost',
        port: '9003',
        pathname: '/face_db_images/**',
      },
    ],
  },
};

export default nextConfig;
