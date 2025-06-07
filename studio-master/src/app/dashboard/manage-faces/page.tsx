import ManageFacesClient from '@/components/dashboard/ManageFacesClient';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Users } from 'lucide-react';

export default function ManageFacesPage() {
  return (
    <div className="container mx-auto py-8">
      <Card className="w-full">
        <CardHeader>
          <CardTitle className="text-2xl font-headline flex items-center">
            <Users className="w-7 h-7 mr-3 text-primary" />
            Face Database Management
          </CardTitle>
        </CardHeader>
        <CardContent>
          <ManageFacesClient />
        </CardContent>
      </Card>
    </div>
  );
}
