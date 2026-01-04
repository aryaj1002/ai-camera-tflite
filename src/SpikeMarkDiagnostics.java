package org.firstinspires.ftc.teamcode.Tests;

import com.qualcomm.robotcore.eventloop.opmode.Disabled;
import com.qualcomm.robotcore.eventloop.opmode.LinearOpMode;
import com.qualcomm.robotcore.eventloop.opmode.TeleOp;
import com.qualcomm.robotcore.util.ElapsedTime;

import org.firstinspires.ftc.teamcode.Hardware.CenterStageBot;
import org.firstinspires.ftc.teamcode.Hardware.CenterStageBotV2;
import org.firstinspires.ftc.teamcode.HardwareBase;
import org.firstinspires.ftc.teamcode.Pipelines.ClassifyAndSaveImagePipeline;
import org.openftc.easyopencv.OpenCvCamera;

@Disabled
@TeleOp(name = "Spike Mark Diagnostic")
public class SpikeMarkDiagnostics extends LinearOpMode {
    // TeleOp class to run the TensorFlow ML model to detect on
    // which spike mark our team's game scoring element (red/blue cyberpunk hat)

    public OpenCvCamera webcam;
    HardwareBase robot = new CenterStageBotV2();
    ElapsedTime runtime = new ElapsedTime();

    public void runOpMode() {

        robot.init(this, hardwareMap, false, true, false, true, runtime);
        ClassifyAndSaveImagePipeline sip = new ClassifyAndSaveImagePipeline();
        sip.setContext(hardwareMap.appContext);
        sip.setResetAfterPic(true);
        sip.setTakePic(false);
        robot.webcam.setPipeline(sip);

        while (!isStarted()) {
            sleep(50);
            telemetry.addData("ML", sip.getAnalysis());
            telemetry.update();
        }

        while (opModeIsActive()) {
            if (gamepad1.left_trigger > 0.1) {
                sip.setTakePic(true);
            }
            telemetry.addData("ML", sip.getAnalysis());
            telemetry.update();
        }
        sip.shutdown();
    }
}
