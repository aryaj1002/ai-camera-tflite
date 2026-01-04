package org.firstinspires.ftc.teamcode.Pipelines;


import android.content.Context;
import android.graphics.Bitmap;
import android.os.SystemClock;
import android.util.Log;

import org.firstinspires.ftc.teamcode.TankDrive;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.imgproc.Imgproc;
import org.openftc.easyopencv.OpenCvPipeline;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.InterpreterApi;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.nio.MappedByteBuffer;
import java.text.SimpleDateFormat;
import java.util.Date;

public class ClassifyAndSaveImagePipeline extends OpenCvPipeline {
    private static final String TAG = "Clsfy&SaveImgPipeline";
    private static final String ASSET_MODEL_FILENAME = "model.tflite";
    private static final float MIN_SECONDS_BETWEEN_PICTURES = 1.0f;
    private boolean takePic = true;
    private boolean resetAfterPic = false;
    private InterpreterApi tflite;
    private Date lastPicTaken;
    private int countPicsTaken = 0;
    private int inferredClass = 0;
    private float inferredConfidence = -1.0f;


    public void setContext(Context context) {
        Log.d(TAG, "setContext starting");
        try {
            MappedByteBuffer tfliteModel = FileUtil.loadMappedFile(context, ASSET_MODEL_FILENAME);
            this.tflite = InterpreterApi.create(tfliteModel, new InterpreterApi.Options());
        } catch (Exception ex) {
            Log.d(TAG, "Issue loading TF model");
            Log.e(TAG, ex.getMessage());
        }
        Log.d(TAG, "setContext finished");
    }

    public void shutdown() {
        if (this.tflite != null) {
            try {
                InterpreterApi temp = this.tflite;
                this.tflite = null;
                temp.close();
            } catch (Exception ex) {
                Log.i(TAG, "tflite close issue");
                Log.i(TAG, ex.getMessage());
            }
        }
    }

    public void setTakePic(boolean takePic) {
        this.takePic = takePic;
    }

    public void setResetAfterPic(boolean resetAfterPic) { this.resetAfterPic = resetAfterPic; }

    public int getInferredClass() {
        return inferredClass;
    }

    public float getInferredConfidence() {
        return inferredConfidence;
    }

    public String getAnalysis() {
        return String.format("Class %d (%.2f%%) PC: %d", this.inferredClass, this.inferredConfidence * 100, this.countPicsTaken);
    }

    private void recognize(Mat input) {
        // if you change your model, you can use a tool like https://netron.app/ to figure out
        // what the input parameter shape(s)/type(s) are, as well as that of the output.

        //  0 <= _rowRange.start && _rowRange.start <= _rowRange.end && _rowRange.end <= m.rows
        Log.i(TAG,input.cols() + " , " +  input.rows() ); // 800 , 448
        Rect rect = new Rect(0, 225, input.cols(), input.rows()-225);
        Mat cropedMat = new Mat(input, rect);

        //Mat destination = new Mat();
        //Imgproc.cvtColor(cropedMat, destination, Imgproc.COLOR_BGR2RGB); // maybe we should train the model in BGR to save this step

        Bitmap bitmap = Bitmap.createBitmap(cropedMat.cols(), cropedMat.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(cropedMat, bitmap);

        ImageProcessor imageProcessor =
                new ImageProcessor.Builder()
                        .add(new ResizeOp(200, 200, ResizeOp.ResizeMethod.BILINEAR))
                        .build();

        // Create a TensorImage object. This creates the tensor of the corresponding
        // tensor type (FLOAT32 in this case) that the TensorFlow Lite interpreter needs.
        TensorImage tensorImage = new TensorImage(DataType.FLOAT32);

        // Analysis code for every frame
        // Preprocess the image
        tensorImage.load(bitmap);
        tensorImage = imageProcessor.process(tensorImage);

        TensorBuffer probabilityBuffer = TensorBuffer.createFixedSize(new int[]{1, 3}, DataType.FLOAT32);
        //TensorBufferFloat probabilityBuffer = (TensorBufferFloat) TensorBufferFloat.createFixedSize(new int[]{1,3}, DataType.FLOAT32);
        long startTime = SystemClock.uptimeMillis();
        try {
            this.tflite.run(tensorImage.getBuffer(), probabilityBuffer.getBuffer());

        } catch (Exception ex) {
            Log.d(TAG, "problem with TF lite run");
            Log.e(TAG, ex.getMessage());
        }

        long endTime = SystemClock.uptimeMillis();

        float[] classifications = probabilityBuffer.getFloatArray();

        // below is softmax func, probably not needed if model outputs softmax
        int maxPos = 0;
        double maxConfidence = 0;
        double sumExp = 0;
        for (int i = 0; i < classifications.length; i++) {
            double exp = Math.exp(classifications[i]);
            sumExp += exp;
            if (classifications[i] > maxConfidence) {
                maxConfidence = exp;
                maxPos = i;
            }
        }
        this.inferredClass = maxPos + 1; // index started at zero, so add one
        this.inferredConfidence = (float) (maxConfidence / sumExp);
        Log.d(TAG, String.format("CL: %d (%.2f%%) ET: %dms IT: %dns",
                this.inferredClass, this.inferredConfidence * 100, (endTime - startTime),
                this.tflite.getLastNativeInferenceDurationNanoseconds()));
    }

    @Override
    public Mat processFrame(Mat input) {

        if (this.tflite != null) {
            try {
                recognize(input);
            } catch (Exception ex)
            {
                Log.e(TAG, ex.getMessage());
            }
        }
        if (takePic) {
            if (resetAfterPic) {
                takePic = false;
            }
            Date date = new Date();
            if (this.lastPicTaken == null) {
                this.lastPicTaken = date;
            }
            long diffInMillies = date.getTime() - this.lastPicTaken.getTime();
            // Log.d(TAG, "Pic time diff = " + diffInMillies);
            if (diffInMillies / 1000 > MIN_SECONDS_BETWEEN_PICTURES) {
                this.lastPicTaken = date;
                SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd-HH-mm-ss");

                String filename = String.format("Bot-C%d-%.0f-%s",
                        this.inferredClass, this.inferredConfidence * 100, dateFormat.format(date));

                this.saveMatToDisk(input, filename);
                this.countPicsTaken++;
                Log.i(TAG, "Pic " + this.countPicsTaken + " saved as " + filename);
            }
            return input;
        } else {
            return input;
        }
    }
}
