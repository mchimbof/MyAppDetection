package com.example.myappdetection;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.drawable.BitmapDrawable;
import android.os.Bundle;
import android.Manifest;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.example.myappdetection.ml.ModelUnquant;
import com.example.myappdetection.ml.Modeloflowers;
import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import org.w3c.dom.Text;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.text.DecimalFormat;
import java.util.List;

public class MainActivity extends AppCompatActivity
        implements OnSuccessListener<Text>, OnFailureListener {
    public static int REQUEST_CAMERA = 111;
    public static int REQUEST_GALLERY = 222;
    Bitmap mSelectedImage;
    ImageView mImageView;
    TextView txtResults;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        //Permiso de ejecucion
        if(checkSelfPermission(Manifest.permission.CAMERA) !=
                PackageManager.PERMISSION_GRANTED)
            requestPermissions(new String[]{Manifest.permission.CAMERA},100);

        //Referencia TextView y ImageView
        mImageView = findViewById(R.id.image_view);
        txtResults = findViewById(R.id.txtresults);
    }
    //Carga los archivos a la GUI
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK && null != data) {
            try {
                if (requestCode == REQUEST_CAMERA)
                    mSelectedImage = (Bitmap) data.getExtras().get("data");
                else
                    mSelectedImage = MediaStore.Images.Media.getBitmap(getContentResolver(), data.getData());
                mImageView.setImageBitmap(mSelectedImage);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
    }
    public void abrirGaleria (View view){
        Intent i = new Intent(Intent.ACTION_PICK,
                android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        startActivityForResult(i, REQUEST_GALLERY);
    }
    public void abrirCamera (View view){
        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        startActivityForResult(intent, REQUEST_CAMERA);
    }
    public void Rostrosfx(View v) {
        InputImage image = InputImage.fromBitmap(mSelectedImage, 0);
        FaceDetectorOptions options =
                new FaceDetectorOptions.Builder()
                        .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
                        .setContourMode(FaceDetectorOptions.CONTOUR_MODE_ALL)
                        .build();
        FaceDetector detector = FaceDetection.getClient(options);
        detector.process(image)
                .addOnSuccessListener(new OnSuccessListener<List<Face>>() {
                    @Override
                    public void onSuccess(List<Face> faces) {
                        if (faces.size() == 0) {
                            Toast.makeText(getApplicationContext(),"No existe rostro",Toast.LENGTH_SHORT).show();
                        }
                        //Colorea los rostros
                        BitmapDrawable drawable = (BitmapDrawable) mImageView.getDrawable();
                        Bitmap bitmap = drawable.getBitmap().copy(Bitmap.Config.ARGB_8888,true);
                        Canvas canvas = new Canvas(bitmap);
                        Paint paint = new Paint();
                        paint.setColor(Color.GREEN);
                        paint.setStrokeWidth(5);
                        paint.setStyle(Paint.Style.STROKE);
                        for (Face rostro:faces) {
                            canvas.drawRect(rostro.getBoundingBox(), paint);
                        }
                        mImageView.setImageBitmap(bitmap);
                    }
                })
                .addOnFailureListener(this);
    }

    public void PersonalizedModel(View v){
        try {
            String[] etiquetas = {"centeno", "chimbo",  "arraes"};
            ModelUnquant model = ModelUnquant.newInstance(getApplicationContext());
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
            inputFeature0.loadBuffer(convertirImagenATensorBuffer(mSelectedImage));
            ModelUnquant.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
            txtResults.setText(obtenerEtiquetayProbabilidad(etiquetas, outputFeature0.getFloatArray()));
            model.close();
        } catch (Exception e) {
            txtResults.setText(e.getMessage());
        }
    }
    public ByteBuffer convertirImagenATensorBuffer(Bitmap mSelectedImage){
        Bitmap imagen = Bitmap.createScaledBitmap(mSelectedImage, 224, 224, true);
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * 224 * 224 * 3);
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues = new int[224 * 224];
        imagen.getPixels(intValues, 0, imagen.getWidth(), 0, 0, imagen.getWidth(), imagen.getHeight());
        int pixel = 0;
        for(int i = 0; i < imagen.getHeight(); i ++){
            for(int j = 0; j < imagen.getWidth(); j++){
                int val = intValues[pixel++]; // RGB
                byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255.f));
                byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255.f));
                byteBuffer.putFloat((val & 0xFF) * (1.f / 255.f));
            }
        }
        return byteBuffer;
    }
    public String obtenerEtiquetayProbabilidad(String[] etiquetas, float[] probabilidades){

        Rostrosfx(null);

        float valorMayor=Float.MIN_VALUE;
        int pos=-1;
        for (int i = 0; i < probabilidades.length; i++) {
            if (probabilidades[i] > valorMayor) {
                valorMayor = probabilidades[i];
                pos = i;
            }
        }
        return "Predicción: " + etiquetas[pos] + ", \nProbabilidad: " +
                (new DecimalFormat("0.00").format(probabilidades[pos] * 100)) + "%";
    }

    @Override
    public void onFailure(@NonNull Exception e) {
        txtResults.setText("Error al procesar imagen");
    }
    @Override
    public void onSuccess(Text text) {}
}