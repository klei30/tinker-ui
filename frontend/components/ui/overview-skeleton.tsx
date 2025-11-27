import { Card, CardContent, CardHeader } from '@/components/ui/card';

export function OverviewSkeleton() {
  return (
    <div className="space-y-8">
      {/* Stats skeleton */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <Card key={i}>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <div className="h-4 bg-muted animate-pulse rounded w-20"></div>
              <div className="h-4 w-4 bg-muted animate-pulse rounded"></div>
            </CardHeader>
            <CardContent>
              <div className="h-8 bg-muted animate-pulse rounded w-16 mb-1"></div>
              <div className="h-3 bg-muted animate-pulse rounded w-24"></div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Cards skeleton */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {Array.from({ length: 4 }).map((_, i) => (
          <Card key={i}>
            <CardHeader className="text-center">
              <div className="mx-auto w-12 h-12 bg-muted animate-pulse rounded-full mb-2"></div>
              <div className="h-5 bg-muted animate-pulse rounded w-24 mx-auto mb-2"></div>
              <div className="h-4 bg-muted animate-pulse rounded w-32 mx-auto"></div>
            </CardHeader>
            <CardContent className="text-center">
              <div className="h-9 bg-muted animate-pulse rounded w-24 mx-auto"></div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Getting started skeleton */}
      <Card>
        <CardHeader>
          <div className="h-6 bg-muted animate-pulse rounded w-32"></div>
          <div className="h-4 bg-muted animate-pulse rounded w-48 mt-2"></div>
        </CardHeader>
        <CardContent className="space-y-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="flex items-start gap-4">
              <div className="w-8 h-8 bg-muted animate-pulse rounded-full"></div>
              <div className="flex-1">
                <div className="h-5 bg-muted animate-pulse rounded w-32 mb-1"></div>
                <div className="h-4 bg-muted animate-pulse rounded w-64"></div>
              </div>
            </div>
          ))}
        </CardContent>
      </Card>
    </div>
  );
}