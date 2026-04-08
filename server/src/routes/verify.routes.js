import { Router } from "express";
import axios from "axios";
import { storageService } from "../services/storageService.js";
import { getIo } from "../socket.js";

const router = Router();

router.post("/", async (request, response) => {
  try {
    const { text, articleId, source, title, url } = request.body;

    if (!text?.trim()) {
      return response.status(400).json({ message: "Text is required for verification." });
    }

    const mlResponse = await axios.post(
      `${process.env.ML_SERVICE_URL || "http://127.0.0.1:5001"}/predict`,
      { text, source, url, title }
    );

    const result = {
      label: mlResponse.data.label,
      confidence: mlResponse.data.confidence
    };

    await storageService.applyVerification(articleId, url, result);

    const record = await storageService.saveVerification({
      newsId: articleId || undefined,
      title: title || text.slice(0, 120),
      source: source || "Custom Search",
      url: url || "",
      text,
      label: result.label,
      confidence: result.confidence
    });

    getIo().emit("verify:created", record);
    return response.json({ result });
  } catch (error) {
    const message =
      error.response?.data?.message ||
      error.message ||
      "Verification service is unavailable.";
    return response.status(500).json({ message });
  }
});

export default router;
