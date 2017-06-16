using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;
using System.Threading;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.Util;
using Emgu.CV.VideoSurveillance;
using Emgu.CV.Cvb;
using System.Runtime.InteropServices;

namespace CameraCapture
{
    public partial class CameraCapture : Form
    {
        [DllImport("user32.dll", CharSet = CharSet.Auto, CallingConvention = CallingConvention.StdCall)]
        //[DllImport("user32.dll")]
        public static extern void mouse_event(long dwFlags, long dx, long dy, long cButtons, long dwExtraInfo);
        private const uint MOUSEEVENTF_MOVE = 0x0001;
        private const uint MOUSEEVENTF_LEFTDOWN = 0x0002;
        private const uint MOUSEEVENTF_LEFTUP = 0x0004;
        private const uint MOUSEEVENTF_MIDDLEDOWN = 0x0020;
        private const uint MOUSEEVENTF_MIDDLEUP = 0x0040;
        private const uint MOUSEEVENTF_RIGHTDOWN = 0x0008;
        private const uint MOUSEEVENTF_RIGHTUP = 0x0010;
        private const uint MOUSEEVENTF_ABSOLUTE = 0x8000;
        private const uint MOUSEEVENTF_VWHEEL = 0x0800;
        private const uint MOUSEEVENTF_HWHEEL = 0x1000;
        private Capture capture = null;
        private bool _captureInProgress;
        private static BackgroundSubtractor fgDetector;
        private static Emgu.CV.Cvb.CvBlobDetector blobDetector;
        private int minarea = 400;
        private int maxarea = 3000;
        private int redThres = 65;
        private int blueThres = 100;
        private int greenThres = 22;
        private int mouseflag = 0;
        private int ccount = 0;
        private int scroll_x = 0;
        private int scroll_y = 0;
        private int scroll_mul_h = 5;
        private int scroll_mul_v = 5;
        private int safevalue = 5;
        private double cursor_mul = 1.5;


        public CameraCapture()
        {
            InitializeComponent();
            CvInvoke.UseOpenCL = false;
            try
            {
                capture = new Capture();
                capture.ImageGrabbed += ProcessFrame;
            }
            catch (NullReferenceException excpt)
            {
                MessageBox.Show(excpt.Message);
            }

            fgDetector = new Emgu.CV.VideoSurveillance.BackgroundSubtractorMOG2();
            blobDetector = new Emgu.CV.Cvb.CvBlobDetector();
        }

        private void ProcessFrame(object sender, EventArgs arg)
        {

            Mat frame = new Mat();
            capture.Retrieve(frame, 0);
            Mat grayFrame = new Mat();
            CvInvoke.CvtColor(frame, grayFrame, ColorConversion.Bgr2Gray);
            //Mat smallGrayFrame = new Mat();
            //CvInvoke.PyrDown(grayFrame, smallGrayFrame);
            //Mat smoothedGrayFrame = new Mat();
            //CvInvoke.PyrUp(smallGrayFrame, smoothedGrayFrame);

            //Image<Gray, Byte> smallGrayFrame = grayFrame.PyrDown();
            //Image<Gray, Byte> smoothedGrayFrame = smallGrayFrame.PyrUp();
            //Mat cannyFrame = new Mat();
            //CvInvoke.Canny(smoothedGrayFrame, cannyFrame, 100, 60);

            //Image<Gray, Byte> cannyFrame = smoothedGrayFrame.Canny(100, 60);
            Image<Bgra, Byte> _frame = frame.ToImage<Bgra, Byte>();
            Image<Gray, Byte> _grayFrame = grayFrame.ToImage<Gray, Byte>();
            Image<Gray, Byte>[] rgb_frame = _frame.Split();              //components of rgb image
            Image<Gray, Byte> red_com = rgb_frame[2] - _grayFrame;
            var red_bi = red_com.Convert<Gray, byte>().ThresholdBinary(new Gray(redThres), new Gray(255));
            Image<Gray, Byte> blue_com = rgb_frame[0] - _grayFrame;
            var blue_bi = blue_com.Convert<Gray, byte>().ThresholdBinary(new Gray(blueThres), new Gray(255));
            Image<Gray, Byte> green_com = rgb_frame[1] - _grayFrame;
            var green_bi = green_com.Convert<Gray, byte>().ThresholdBinary(new Gray(greenThres), new Gray(255));
            //System.Windows.Forms.MessageBox.Show("");



            ///////////////////////////////////////////////////////////////////////////////////
            //Blob detection

            //Red Blob detection
            Image<Bgr, Byte> smoothedFrame_r = new Image<Bgr, byte>(red_com.Size);
            CvInvoke.GaussianBlur(red_bi, smoothedFrame_r, new Size(3, 3), 1); //filter out noises

            Mat forgroundMask_r = new Mat();
            fgDetector.Apply(smoothedFrame_r, forgroundMask_r);

            CvBlobs blobs_r = new CvBlobs();
            blobDetector.Detect(forgroundMask_r.ToImage<Gray, byte>(), blobs_r);
            blobs_r.FilterByArea(minarea, maxarea);


            //blue Blob Detection
            Image<Bgr, Byte> smoothedFrame_b = new Image<Bgr, byte>(red_com.Size);
            CvInvoke.GaussianBlur(blue_bi, smoothedFrame_b, new Size(3, 3), 1); //filter out noises

            Mat forgroundMask_b = new Mat();
            fgDetector.Apply(smoothedFrame_b, forgroundMask_b);

            CvBlobs blobs_b = new CvBlobs();
            blobDetector.Detect(forgroundMask_b.ToImage<Gray, byte>(), blobs_b);
            blobs_b.FilterByArea(minarea, maxarea);


            //Green blob detection
            Image<Bgr, Byte> smoothedFrame_g = new Image<Bgr, byte>(red_com.Size);
            CvInvoke.GaussianBlur(green_bi, smoothedFrame_g, new Size(3, 3), 1); //filter out noises

            Mat forgroundMask_g = new Mat();
            fgDetector.Apply(smoothedFrame_g, forgroundMask_g);

            CvBlobs blobs_g = new CvBlobs();
            blobDetector.Detect(forgroundMask_g.ToImage<Gray, byte>(), blobs_g);
            blobs_g.FilterByArea(minarea, maxarea);


            //Mouse Interpretition
            float[] cent_r = new float[2];
            float[] cent_g = new float[2];
            float[] cent_b = new float[2];
            //Corsor control with Green Marker
            foreach (var pair in blobs_g)
            {
                CvBlob b = pair.Value;
                CvInvoke.Rectangle(frame, b.BoundingBox, new MCvScalar(255.0, 255.0, 255.0), 2);
                cent_g[0] = b.Centroid.X;
                cent_g[1] = b.Centroid.Y;
            }
            if (blobs_g.Count == 1 || mouseflag != 0)
            {
                //Cursor Movement Controlled
                //Primary Screem
                //if (Screen.AllScreens.Length == 1)
                {
                    Cursor.Position = new Point(Screen.PrimaryScreen.Bounds.Width - (int)(cursor_mul * (int)cent_g[0] * Screen.PrimaryScreen.Bounds.Width / capture.Width), (int)(cursor_mul * (int)cent_g[1]) * Screen.PrimaryScreen.Bounds.Height / capture.Height);
                }
                //Secondary Screen
                //Cursor.Position = new Point((int)(cursor_mul * (int)cent_g[0] * Screen.AllScreens[1].Bounds.Width / capture.Width), (int)(cursor_mul * (int)cent_g[1]) * Screen.AllScreens[1].Bounds.Height / capture.Height);
                //Number of Screen = 2 and both a same time
                /*   if (Screen.AllScreens.Length == 2)
                   {

                       Cursor.Position = new Point((int)(cursor_mul * (int)cent_g[0] * (Screen.AllScreens[1].Bounds.Width + Screen.AllScreens[0].Bounds.Width) / capture.Width),
                                               (int)(cursor_mul * (int)cent_g[1]) * (Screen.AllScreens[1].Bounds.Height + Screen.AllScreens[0].Bounds.Height) / capture.Height);
                   }
                   //Number of screen =3 and all at same time
                   if (Screen.AllScreens.Length == 3)
                   {

                       Cursor.Position = new Point((int)(cursor_mul * (int)cent_g[0] * (Screen.AllScreens[1].Bounds.Width + Screen.AllScreens[0].Bounds.Width + Screen.AllScreens[2].Bounds.Width) / capture.Width),
                                               (int)(cursor_mul * (int)cent_g[1]) * (Screen.AllScreens[1].Bounds.Height + Screen.AllScreens[0].Bounds.Height + Screen.AllScreens[0].Bounds.Height) / capture.Height);
                   }
                       */

                /*
                //Check for Clicks
                if (blobs_r.Count == 1)
                {
                    if(blobs_g.Count == 0)
                    {
                        if(ccount == 1)
                        {
                            //double click
                            mouse_event(MOUSEEVENTF_LEFTDOWN, (int)cent_g[0], (int)cent_g[1], 0, 0);
                            mouse_event(MOUSEEVENTF_LEFTUP, (int)cent_g[0], (int)cent_g[1], 0, 0);
                            Thread.Sleep(150);
                            mouse_event(MOUSEEVENTF_LEFTDOWN, (int)cent_g[0], (int)cent_g[1], 0, 0);
                            mouse_event(MOUSEEVENTF_LEFTUP, (int)cent_g[0], (int)cent_g[1], 0, 0);
                        }
                        else
                        {
                            ccount--;
                        }
                    }

                    else if ((cent_g[0] - cent_r[0] >= 10 || cent_r[0] - cent_g[0] <= 10) && (cent_g[1] - cent_r[1] >= 10 || cent_r[1] - cent_g[1] <= 10))
                    {
                        ccount = safevalue;
                        mouseflag = 1;
                        //single click
                        mouse_event(MOUSEEVENTF_LEFTDOWN, (int)cent_g[0], (int)cent_g[1], 0, 0);
                        mouse_event(MOUSEEVENTF_LEFTUP, (int)cent_g[0], (int)cent_g[1], 0, 0);
                    }
                }
                else
                {
                    ccount = 0;

                }

            }

            if (blobs_b.Count == 1)
                {
                    foreach (var pair in blobs_b)
                    {
                        CvBlob b = pair.Value;
                        CvInvoke.Rectangle(frame, b.BoundingBox, new MCvScalar(255.0, 255.0, 255.0), 2);
                        cent_b[0] = b.Centroid.X;
                        cent_b[1] = b.Centroid.Y;
                    }
                  
                     if (blobs_g.Count == 1 && (cent_g[0] - cent_b[0] >= 10 || cent_b[0] - cent_g[0] <= 10) && (cent_g[1] - cent_b[1] >= 10 || cent_b[1] - cent_g[1] <= 10))
                    {
                        //right click
                        mouse_event(MOUSEEVENTF_RIGHTDOWN, (int)cent_g[0], (int)cent_g[1], 0, 0);
                        mouse_event(MOUSEEVENTF_RIGHTUP, (int)cent_g[0], (int)cent_g[1], 0, 0);
                    }
                  
                    else if(blobs_g.Count == 0)
                    {
                        mouse_event(MOUSEEVENTF_VWHEEL, 0, 0, (scroll_y - (int)cent_b[1]) * scroll_mul_v, 0);
                        mouse_event(MOUSEEVENTF_HWHEEL, 0, 0, (scroll_x - (int)cent_b[0]) * scroll_mul_h, 0);                     
                        scroll_y = (int)cent_b[1];
                        scroll_x = (int)cent_b[0];

                    }
                */
            }


            captureImageBox.Image = frame;
            grayscaleImageBox.Image = red_bi;
            smoothedGrayscaleImageBox.Image = green_bi;
            cannyImageBox.Image = blue_bi;





        }



        private void captureButtonClick(object sender, EventArgs e)
        {
            if (capture != null)
            {
                if (_captureInProgress)
                {  //stop the capture
                    captureButton.Text = "Start Capture";
                    capture.Pause();
                }
                else
                {
                    //start the capture
                    captureButton.Text = "Stop";
                    capture.Start();
                }

                _captureInProgress = !_captureInProgress;
            }
        }


        private void ReleaseData()
        {
            if (capture != null)
                capture.Dispose();
        }

        private void FlipHorizontalButtonClick(object sender, EventArgs e)
        {
            if (capture != null) capture.FlipHorizontal = !capture.FlipHorizontal;
        }

        private void FlipVerticalButtonClick(object sender, EventArgs e)
        {
            if (capture != null) capture.FlipVertical = !capture.FlipVertical;
        }
    }
}
