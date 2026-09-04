# Open questions before HPC execution

These are deliberate execution gates. The scaffold fails explicitly when one
of these facts is missing; it does not invent a processor, split, label, pool,
or metric.

1. Pin exact revisions for all Hugging Face repositories and record them in
   the run metadata. The current final plan names repositories but does not
   provide immutable revisions.
2. Confirm the exact official SSv2 action-label file and its class-index
   ordering. The manifest stores the official action string and accepts an
   external official label map; it does not derive a research label mapping.
3. Register the fixed H1 prompt and the exact H3 prompt-generation output
   before feature extraction. The final plan requires official prompt
   handling but does not state the text/template. Extraction therefore takes
   an external `example_id -> prompt` JSON file and does not compose one.
4. Pin the official InternVL3.5 Hugging Face processor/chat-template path and
   the decoder module paths needed for every language layer and video-token
   position.
5. Pin the official MiniCPM-V 4.5 local inference/processor API and decoder
   module paths needed for the same two readouts.
6. Integrate Meta's official V-JEPA 2 ViT-L frozen-probe preprocessing and
   layer-output path. V-JEPA does not receive a compact state in this protocol.
7. Confirm the exact EgoTempo clip-ID downloader and judge command available on
   the HPC account. Only the requested clip IDs may be downloaded.
8. Confirm the V-STaR official evaluation command and its what/when/where
   result schema. The project must archive that command and raw output rather
   than reimplement a metric.
