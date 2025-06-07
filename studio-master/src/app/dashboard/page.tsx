// in full-stack/studio-master/src/app/dashboard/page.tsx
import LiveFeed from '@/components/dashboard/LiveFeed';
import { Button } from '@/components/ui/button';
import { Users, Upload } from 'lucide-react'; // Import Upload icon
import Link from 'next/link';

export default function DashboardPage() {
  return (
    <div className="space-y-8">
      <div className="flex justify-end gap-4"> {/* Use gap for spacing */}
        <Button asChild variant="outline">
          <Link href="/dashboard/upload-analysis"> {/* New Link */}
            <Upload className="w-4 h-4 mr-2" />
            Upload & Deep Analyze
          </Link>
        </Button>
        <Button asChild variant="outline">
          <Link href="/dashboard/manage-faces">
            <Users className="w-4 h-4 mr-2" />
            Manage Face Database
          </Link>
        </Button>
      </div>
      <div>
        <LiveFeed />
      </div>
    </div>
  );
}