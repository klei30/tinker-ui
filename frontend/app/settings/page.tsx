"use client";

import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { ArrowLeft } from "lucide-react";
import { HuggingFaceSettings } from "@/components/settings/huggingface-settings";

export default function SettingsPage() {
  const router = useRouter();

  return (
    <div className="container py-8">
      <div className="mb-6">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => router.back()}
          className="mb-4"
        >
          <ArrowLeft className="mr-2 h-4 w-4" />
          Back
        </Button>
        <h1 className="text-3xl font-bold">Settings</h1>
        <p className="text-muted-foreground mt-1">
          Configure your integrations and preferences
        </p>
      </div>

      <div className="space-y-6">
        <HuggingFaceSettings />

        {/* Add more settings sections here */}
      </div>
    </div>
  );
}
