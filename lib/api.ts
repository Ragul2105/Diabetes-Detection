/**
 * API service for communicating with the FastAPI backend
 */

// Use environment variable with fallback for the Hugging Face deployed server
//const API_URL = process.env.NEXT_PUBLIC_API_URL || "https://harishvijayasarangan-dr-server.hf.space";
const API_URL = process.env.NEXT_PUBLIC_API_URL || "https://SairamDev-selfie-dr.hf.space";

// Gemini API configuration
const GEMINI_API_KEY = process.env.NEXT_PUBLIC_GEMINI_API_KEY || "";
const GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent";

export interface AnalysisResult {
  detailed_classification: {
    [key: string]: number;
  };
  highest_probability_class: string;
}

export interface GeminiAssessment {
  description: string;
  cause: string;
  remedy: string;
}

/**
 * Generates a specialist assessment using Gemini API
 * @param classification The detected diabetic retinopathy classification
 * @param probability The probability percentage
 * @returns AI-generated specialist description and remedy
 */
export async function getGeminiAssessment(
  classification: string,
  probability: number
): Promise<GeminiAssessment> {
  try {
    const prompt = `You are a diabetic retinopathy specialist. A patient's retinal scan has been analyzed and detected as "${classification}" with ${probability.toFixed(1)}% confidence.

Please provide:
1. A professional explanation of this condition in 3-4 lines, explaining what it means for the patient in simple terms.
2. Potential causes for the infection that may have led to this stage of diabetic retinopathy in 2-3 lines.
3. A single line remedy or recommended next step.

Format your response exactly as:
DESCRIPTION: [Your 3-4 line explanation here]
CAUSE: [Your 2-3 line potential causes here]
REMEDY: [Your one line remedy here]`;

    const response = await fetch(`${GEMINI_API_URL}?key=${GEMINI_API_KEY}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        contents: [
          {
            parts: [
              {
                text: prompt,
              },
            ],
          },
        ],
        generationConfig: {
          temperature: 0.7,
          maxOutputTokens: 500,
        },
      }),
    });

    if (!response.ok) {
      // Log but don't throw - use fallback instead
      console.warn(`Gemini API returned ${response.status}, using fallback assessment`);
      return getFallbackAssessment(classification);
    }

    const data = await response.json();
    const text = data.candidates?.[0]?.content?.parts?.[0]?.text || "";

    // Parse the response
    const descriptionMatch = text.match(/DESCRIPTION:\s*([\s\S]*?)(?=CAUSE:|$)/i);
    const causeMatch = text.match(/CAUSE:\s*([\s\S]*?)(?=REMEDY:|$)/i);
    const remedyMatch = text.match(/REMEDY:\s*([\s\S]*?)$/i);

    return {
      description: descriptionMatch?.[1]?.trim() || "Assessment not available.",
      cause: causeMatch?.[1]?.trim() || "Cause assessment not available.",
      remedy: remedyMatch?.[1]?.trim() || "Please consult with a healthcare professional.",
    };
  } catch (error) {
    console.warn("Gemini API unavailable, using fallback assessment");
    return getFallbackAssessment(classification);
  }
}

/**
 * Returns a fallback assessment when Gemini API is unavailable
 */
function getFallbackAssessment(classification: string): GeminiAssessment {
  const assessments: Record<string, GeminiAssessment> = {
    "No DR": {
      description: "Your retinal scan shows no signs of diabetic retinopathy. The blood vessels in your retina appear healthy without any diabetes-related damage. This is an excellent result, but continued monitoring is important as diabetic retinopathy can develop over time in people with diabetes.",
      cause: "Not applicable - no diabetic retinopathy detected. Your current diabetes management appears to be effectively protecting your eye health.",
      remedy: "Maintain regular annual eye exams and keep your blood sugar levels well controlled."
    },
    "Mild": {
      description: "Your scan indicates mild non-proliferative diabetic retinopathy (NPDR). Small areas of balloon-like swelling called microaneurysms have been detected in the retina's blood vessels. At this early stage, there is typically no noticeable vision loss, but it signals that diabetes is beginning to affect your eyes.",
      cause: "This early stage may be caused by prolonged periods of elevated blood sugar levels, which weaken the tiny blood vessels in the retina. Contributing factors could include inconsistent diabetes management, high blood pressure, or duration of diabetes.",
      remedy: "Schedule a follow-up with your ophthalmologist within 6-12 months and focus on strict blood sugar control."
    },
    "Moderate": {
      description: "Moderate non-proliferative diabetic retinopathy has been detected. The blood vessels in your retina are showing more significant damage, with some vessels becoming blocked. This can lead to reduced blood flow to parts of your retina and may start affecting your vision quality.",
      cause: "Progression to this stage is typically associated with chronic hyperglycemia over several years, combined with factors like uncontrolled hypertension, high cholesterol, or smoking. Inadequate diabetes management accelerates vessel damage.",
      remedy: "Consult your ophthalmologist within 3-6 months for detailed examination and potential treatment planning."
    },
    "Severe": {
      description: "Your scan reveals severe non-proliferative diabetic retinopathy. Many blood vessels in your retina are blocked, depriving several areas of adequate blood supply. This significantly increases the risk of progression to proliferative diabetic retinopathy and potential vision loss.",
      cause: "Severe retinopathy usually results from years of poorly controlled diabetes with persistent high blood sugar, often compounded by hypertension and kidney disease. The extensive blockage indicates significant cumulative damage to retinal blood vessels.",
      remedy: "Seek urgent consultation with a retina specialist within 2-4 weeks for possible laser treatment or injections."
    },
    "Proliferative DR": {
      description: "Proliferative diabetic retinopathy (PDR), the most advanced stage, has been detected. New, abnormal blood vessels are growing on the retina's surface. These fragile vessels can leak blood into the eye and cause retinal detachment, leading to severe vision loss or blindness if untreated.",
      cause: "PDR develops when severe oxygen deprivation triggers abnormal blood vessel growth. This typically results from long-standing diabetes with poor glycemic control, often combined with hypertension, nephropathy, and delayed treatment of earlier retinopathy stages.",
      remedy: "Seek immediate medical attention from a retina specialist for urgent treatment including laser therapy or vitrectomy."
    }
  };

  return assessments[classification] || {
    description: `${classification} has been detected in your retinal scan. This condition affects the blood vessels in the retina and may impact your vision if left untreated. The severity level indicates how much the retina has been affected by diabetes-related changes.`,
    cause: "Diabetic retinopathy is generally caused by prolonged high blood sugar levels damaging the blood vessels in the retina. Contributing factors may include duration of diabetes, blood pressure, cholesterol levels, and overall diabetes management.",
    remedy: "Please consult with an ophthalmologist for a comprehensive eye examination and treatment plan."
  };
}

/**
 * Analyzes a retinal image using the FastAPI backend
 * @param file The image file to analyze
 * @returns Analysis result with detailed classification and highest probability class
 */
export async function analyzeImage(file: File): Promise<AnalysisResult> {
  try {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(`${API_URL}/predict`, {
      method: "POST",
      body: formData,
      // No need to set Content-Type header as browser will set it correctly with boundary for FormData
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(
        errorData.detail || `Server error: ${response.status}`
      );
    }

    return await response.json();
  } catch (error) {
    console.error("Error analyzing image:", error);
    throw error;
  }
}

/**
 * Checks if the FastAPI server is healthy
 * @returns Health status information
 */
export async function checkServerHealth() {
  try {
    const response = await fetch(`${API_URL}/health`);
    return await response.json();
  } catch (error) {
    console.error("Health check failed:", error);
    return { status: "unhealthy", message: "Could not connect to server" };
  }
}
