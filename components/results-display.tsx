"use client";

import { useState, useEffect } from "react";
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  Cell
} from "recharts";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { AlertCircle, Info, Loader2 } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { getGeminiAssessment, GeminiAssessment } from "@/lib/api";

// Define severity levels and their colors
const severityColors = {
  "No DR": "#22c55e", // Green
  "Mild": "#eab308", // Yellow
  "Moderate": "#f97316", // Orange
  "Severe": "#ef4444", // Red
  "Proliferative DR": "#7f1d1d", // Dark Red
};

type SeverityLevel = keyof typeof severityColors;

interface ResultsDisplayProps {
  result: {
    detailed_classification: Record<string, number>;
    highest_probability_class: string;
  };
}

export function ResultsDisplay({ result }: ResultsDisplayProps) {
  const { detailed_classification, highest_probability_class } = result;
  
  // State for Gemini AI assessment
  const [aiAssessment, setAiAssessment] = useState<GeminiAssessment | null>(null);
  const [isLoadingAssessment, setIsLoadingAssessment] = useState(true);
  
  // Fetch Gemini assessment on mount
  useEffect(() => {
    const fetchAssessment = async () => {
      setIsLoadingAssessment(true);
      const maxProbability = detailed_classification[highest_probability_class] || 0;
      const assessment = await getGeminiAssessment(highest_probability_class, maxProbability);
      setAiAssessment(assessment);
      setIsLoadingAssessment(false);
    };
    
    fetchAssessment();
  }, [highest_probability_class, detailed_classification]);
  
  // Consolidate values below 15% into the maximum value
  const consolidateClassification = (classification: Record<string, number>) => {
    const THRESHOLD = 15;
    const entries = Object.entries(classification);
    
    // Find the maximum value entry
    let maxEntry = entries[0];
    for (const entry of entries) {
      if (entry[1] > maxEntry[1]) {
        maxEntry = entry;
      }
    }
    
    // Calculate consolidated values
    const consolidated: Record<string, number> = {};
    let additionalValue = 0;
    
    for (const [name, value] of entries) {
      if (name === maxEntry[0]) {
        // This is the max entry, will add to it later
        consolidated[name] = value;
      } else if (value < THRESHOLD) {
        // Below threshold, set to 0 and add to the max value
        consolidated[name] = 0;
        additionalValue += value;
      } else {
        // Above threshold, keep as is
        consolidated[name] = value;
      }
    }
    
    // Add the consolidated value to the maximum
    consolidated[maxEntry[0]] += additionalValue;
    
    return consolidated;
  };
  
  const consolidatedClassification = consolidateClassification(detailed_classification);
  
  // Format data for chart - show all entries but with consolidated values
  const chartData = Object.entries(consolidatedClassification)
    .map(([name, value]) => ({
      name,
      value: Math.round(value * 100) / 100, // Round to 2 decimal places
    }));
  
  // Get severity info
  const getSeverityInfo = (severity: string) => {
    const infoMap: Record<string, { description: string; action: string }> = {
      "No DR": { 
        description: "No signs of diabetic retinopathy detected.", 
        action: "Continue with regular eye exams as recommended by your doctor." 
      },
      "Mild": { 
        description: "Early stage diabetic retinopathy with small changes in the retina.", 
        action: "Follow up with your eye doctor within 6-12 months." 
      },
      "Moderate": { 
        description: "More significant changes in the retina that may affect vision.", 
        action: "Consult with your eye doctor within 3-6 months." 
      },
      "Severe": { 
        description: "Substantial changes in the retina with risk of vision loss.", 
        action: "Seek medical attention within 1 month." 
      },
      "Proliferative DR": { 
        description: "Advanced stage with abnormal blood vessel growth. High risk of vision loss.", 
        action: "Seek immediate medical attention." 
      },
    };
    
    return infoMap[severity] || { 
      description: "Classification information not available.", 
      action: "Please consult with a healthcare professional." 
    };
  };
  
  const severityInfo = getSeverityInfo(highest_probability_class);
  const severityColor = severityColors[highest_probability_class as SeverityLevel] || "#6b7280";

  return (
    <div className="space-y-6">
      {/* Main result card */}
      <Card className="overflow-hidden border-t-4" style={{ borderTopColor: severityColor }}>
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center justify-between">
            <span>Analysis Result</span>
            <Badge 
              style={{ 
                backgroundColor: severityColor,
                color: ["No DR", "Mild"].includes(highest_probability_class) ? "#000" : "#fff"
              }}
              className="text-sm py-1"
            >
              {highest_probability_class}
            </Badge>
          </CardTitle>
          <CardDescription>
            Diabetic Retinopathy Classification
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Alert className="mb-4">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Important Note</AlertTitle>
            <AlertDescription>
              This is an AI-assisted analysis and should not replace professional medical advice.
              Please consult with a healthcare professional for proper diagnosis.
            </AlertDescription>
          </Alert>
          
          <div className="space-y-2 mb-6">
            <h4 className="font-medium">Specialist Assessment:</h4>
            {isLoadingAssessment ? (
              <div className="flex items-center gap-2 text-muted-foreground">
                <Loader2 className="h-4 w-4 animate-spin" />
                <span>Generating specialist assessment...</span>
              </div>
            ) : (
              <>
                <p className="text-gray-700 leading-relaxed">
                  {aiAssessment?.description} {aiAssessment?.cause}
                </p>
                <p className="text-xs text-amber-600 mt-2 italic">
                  ⚠️ Note: Potential causes mentioned are assumptions based on general patterns. For accurate diagnosis, please consult a doctor.
                </p>
                
                <h4 className="font-medium mt-4">Recommended Remedy:</h4>
                <p className="text-gray-700 font-medium">{aiAssessment?.remedy}</p>
              </>
            )}
          </div>
        </CardContent>
      </Card>
      
      {/* Chart card */}
      <Card>
        <CardHeader>
          <CardTitle>Detailed Classification</CardTitle>
          <CardDescription>
            Probability distribution across severity levels (values below 15% are consolidated into the highest probability class)
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-[300px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={chartData}
                margin={{ top: 5, right: 30, left: 20, bottom: 60 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="name" 
                  angle={-45} 
                  textAnchor="end"
                  tick={{ fontSize: 12 }}
                  height={70}
                />
                <YAxis 
                  label={{ 
                    value: 'Probability (%)', 
                    angle: -90, 
                    position: 'insideLeft',
                    style: { textAnchor: 'middle' }
                  }}
                />
                <Tooltip formatter={(value) => [`${value}%`, 'Probability']} />
                <Bar dataKey="value" name="Probability">
                  {chartData.map((entry, index) => (
                    <Cell 
                      key={`cell-${index}`} 
                      fill={severityColors[entry.name as SeverityLevel] || "#6b7280"} 
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
