"use client"

import { useState } from "react"
import { Calculator, Info, Sparkles } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Alert, AlertDescription } from "@/components/ui/alert"

interface HyperparamRecommendations {
  learning_rate: number
  batch_size: number
  lora_rank: number
  adam_beta1: number
  adam_beta2: number
  adam_eps: number
}

interface Explanation {
  learning_rate: string
  batch_size: string
  lora_rank: string
  notes: string[]
  source: string
}

interface HyperparamResponse {
  success: boolean
  model_name: string
  recipe_type: string
  recommendations: HyperparamRecommendations
  explanation: Explanation
}

interface HyperparamCalculatorProps {
  modelName: string
  recipeType: string
  loraRank?: number
  onApply: (recommendations: HyperparamRecommendations) => void
}

export function HyperparamCalculator({
  modelName,
  recipeType,
  loraRank,
  onApply
}: HyperparamCalculatorProps) {
  const [recommendations, setRecommendations] = useState<HyperparamResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const calculateRecommendations = async () => {
    if (!modelName) {
      setError("Please select a model first")
      return
    }

    setLoading(true)
    setError(null)

    try {
      // Call backend API directly
      const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
      const response = await fetch(`${API_BASE}/hyperparameters/calculate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_name: modelName,
          recipe_type: recipeType,
          lora_rank: loraRank
        })
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || 'Failed to calculate hyperparameters')
      }

      const data = await response.json()
      setRecommendations(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-4">
      <Button
        onClick={calculateRecommendations}
        disabled={loading || !modelName}
        className="w-full"
        variant="outline"
      >
        <Calculator className="h-4 w-4 mr-2" />
        {loading ? "Calculating..." : "Calculate Recommended Values"}
      </Button>

      {error && (
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {recommendations && (
        <Card className="border-2 border-primary/20 bg-gradient-to-br from-background to-muted/20">
          <CardHeader>
            <div className="flex items-center gap-2">
              <Sparkles className="h-5 w-5 text-primary" />
              <CardTitle>Recommended Hyperparameters</CardTitle>
            </div>
            <CardDescription>
              Based on <span className="font-semibold">{recommendations.model_name}</span> architecture and <span className="font-semibold">{recommendations.recipe_type.toUpperCase()}</span> recipe
            </CardDescription>
          </CardHeader>

          <CardContent className="space-y-4">
            {/* Learning Rate */}
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-sm font-medium">Learning Rate</span>
                <Badge variant="secondary" className="font-mono text-sm">
                  {recommendations.recommendations.learning_rate.toExponential(2)}
                </Badge>
              </div>
              <div className="flex items-start gap-2 text-xs text-muted-foreground bg-muted/50 p-2 rounded">
                <Info className="h-3 w-3 mt-0.5 flex-shrink-0" />
                <span>{recommendations.explanation.learning_rate}</span>
              </div>
            </div>

            {/* Batch Size */}
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-sm font-medium">Batch Size</span>
                <Badge variant="secondary" className="font-mono text-sm">
                  {recommendations.recommendations.batch_size}
                </Badge>
              </div>
              <div className="flex items-start gap-2 text-xs text-muted-foreground bg-muted/50 p-2 rounded">
                <Info className="h-3 w-3 mt-0.5 flex-shrink-0" />
                <span>{recommendations.explanation.batch_size}</span>
              </div>
            </div>

            {/* LoRA Rank */}
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-sm font-medium">LoRA Rank</span>
                <Badge variant="secondary" className="font-mono text-sm">
                  {recommendations.recommendations.lora_rank}
                </Badge>
              </div>
              <div className="flex items-start gap-2 text-xs text-muted-foreground bg-muted/50 p-2 rounded">
                <Info className="h-3 w-3 mt-0.5 flex-shrink-0" />
                <span>{recommendations.explanation.lora_rank}</span>
              </div>
            </div>

            {/* Adam Optimizer */}
            <div className="space-y-2 pt-2 border-t">
              <span className="text-sm font-medium">Adam Optimizer</span>
              <div className="grid grid-cols-3 gap-2 text-xs">
                <div className="flex flex-col">
                  <span className="text-muted-foreground">Beta1</span>
                  <span className="font-mono">{recommendations.recommendations.adam_beta1}</span>
                </div>
                <div className="flex flex-col">
                  <span className="text-muted-foreground">Beta2</span>
                  <span className="font-mono">{recommendations.recommendations.adam_beta2}</span>
                </div>
                <div className="flex flex-col">
                  <span className="text-muted-foreground">Epsilon</span>
                  <span className="font-mono">{recommendations.recommendations.adam_eps.toExponential(1)}</span>
                </div>
              </div>
            </div>

            {/* Important Notes */}
            <div className="space-y-2 pt-2 border-t">
              <span className="text-sm font-medium">Important Notes</span>
              <ul className="space-y-1 text-xs text-muted-foreground">
                {recommendations.explanation.notes.map((note, idx) => (
                  <li key={idx} className="flex items-start gap-2">
                    <span className="text-primary">•</span>
                    <span>{note}</span>
                  </li>
                ))}
              </ul>
            </div>

            {/* Source */}
            <div className="text-xs text-muted-foreground pt-2 border-t">
              <span>Source: </span>
              <a
                href={recommendations.explanation.source}
                target="_blank"
                rel="noopener noreferrer"
                className="text-primary hover:underline"
              >
                Tinker Documentation
              </a>
            </div>
          </CardContent>

          <CardFooter>
            <Button
              onClick={() => onApply(recommendations.recommendations)}
              className="w-full"
            >
              <Sparkles className="h-4 w-4 mr-2" />
              Apply These Values
            </Button>
          </CardFooter>
        </Card>
      )}
    </div>
  )
}
