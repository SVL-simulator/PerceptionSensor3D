/**
 * Copyright (c) 2019-2021 LG Electronics, Inc.
 *
 * This software contains code licensed as described in LICENSE.
 *
 */

using System;
using System.Collections.Generic;
using UnityEngine;
using Simulator.Bridge;
using Simulator.Bridge.Data;
using Simulator.Map;
using Simulator.Utilities;
using Simulator.Sensors.UI;

namespace Simulator.Sensors
{
    using Components;
    using UnityEngine.Experimental.Rendering;
    using UnityEngine.Rendering;
    using UnityEngine.Rendering.HighDefinition;

    [SensorType("3D Ground Truth", new[] { typeof(Detected3DObjectData) })]
    public class PerceptionSensor3D : FrequencySensorBase
    {
        private static class Properties
        {
            public static readonly int TexSize = Shader.PropertyToID("_TexSize");
            public static readonly int Input = Shader.PropertyToID("_Input");
            public static readonly int MaskBuffer = Shader.PropertyToID("_MaskBuffer");
        }

        private class DetectedObject
        {
            public Detected3DObject object3d;
            public Transform transform;
            public Bounds bounds;
        }

        [SensorParameter]
        [Range(1f, 1000f)]
        public float MaxDistance = 100.0f;

        private WireframeBoxes WireframeBoxes;

        private BridgeInstance Bridge;
        private Publisher<Detected3DObjectData> Publish;

        private List<DetectedObject> DetectedObjects = new List<DetectedObject>();

        [AnalysisMeasurement(MeasurementType.Count)]
        public int MaxTracked = -1;

        public ComputeShader cs;

        public override SensorDistributionType DistributionType => SensorDistributionType.MainOrClient;
        public override float PerformanceLoad { get; } = 0.2f;
        private MapOrigin MapOrigin;

        private IAgentController Controller;

        private ShaderTagId passId;
        private Camera sensorCamera;
        private ComputeBuffer maskBuffer;
        private uint[] maskBufferArr = new uint[256];
        private const int CubemapSize = 1024;
        private RTHandle rtColor;
        private RTHandle rtDepth;
        private uint seqId;

        private Camera SensorCamera
        {
            get
            {
                if (sensorCamera == null)
                    sensorCamera = GetComponent<Camera>();

                return sensorCamera;
            }
        }
        
        protected override bool UseFixedUpdate => false;
        
        public override void OnBridgeSetup(BridgeInstance bridge)
        {
            Bridge = bridge;
            Publish = Bridge.AddPublisher<Detected3DObjectData>(Topic);
        }

        protected override void Initialize()
        {
            Controller = GetComponentInParent<IAgentController>();

            MapOrigin = MapOrigin.Find();

            passId = new ShaderTagId("SimulatorSegmentationPass");
            SensorCamera.farClipPlane = MaxDistance;
            var hdData = SensorCamera.gameObject.GetComponent<HDAdditionalCameraData>();
            hdData.customRender += OnSegmentationRender;
            hdData.hasPersistentHistory = true;

            rtColor = RTHandles.Alloc(
                CubemapSize,
                CubemapSize,
                6,
                DepthBits.None,
                GraphicsFormat.R8G8B8A8_UNorm,
                dimension: TextureDimension.Tex2DArray,
                useDynamicScale: false,
                name: "Perc3D_Tex2DArr",
                wrapMode: TextureWrapMode.Clamp);

            rtDepth = RTHandles.Alloc(
                CubemapSize,
                CubemapSize,
                6,
                DepthBits.Depth32,
                GraphicsFormat.R32_UInt,
                dimension: TextureDimension.Tex2DArray,
                useDynamicScale: false,
                name: "Perc3D_Tex2DArr",
                wrapMode: TextureWrapMode.Clamp);

            maskBuffer = new ComputeBuffer(256, sizeof(uint));

            WireframeBoxes = SimulatorManager.Instance.WireframeBoxes;
        }

        protected override void Deinitialize()
        {
            DetectedObjects.Clear();
            
            rtColor?.Release();
            rtDepth?.Release();
            maskBuffer?.Release();
        }

        private void OnSegmentationRender(ScriptableRenderContext context, HDCamera hdCamera)
        {
            var cmd = CommandBufferPool.Get();
            RenderToTextureArray(context, cmd, hdCamera);

            var clearKernel = cs.FindKernel("Clear");
            cmd.SetComputeBufferParam(cs, clearKernel, Properties.MaskBuffer, maskBuffer);
            cmd.DispatchCompute(cs, clearKernel, 4, 1, 1);

            var detectKernel = cs.FindKernel("Detect");
            cmd.SetComputeTextureParam(cs, detectKernel, Properties.Input, rtColor, 0);
            cmd.SetComputeVectorParam(cs, Properties.TexSize, new Vector4(CubemapSize, CubemapSize, 1f / CubemapSize, 1f / CubemapSize));
            cmd.SetComputeBufferParam(cs, detectKernel, Properties.MaskBuffer, maskBuffer);
            cmd.DispatchCompute(cs, detectKernel, HDRPUtilities.GetGroupSize(CubemapSize, 8), HDRPUtilities.GetGroupSize(CubemapSize, 8), 6);

            context.ExecuteCommandBuffer(cmd);
            cmd.Clear();
            CommandBufferPool.Release(cmd);
        }

        private void RenderToTextureArray(ScriptableRenderContext context, CommandBuffer cmd, HDCamera hd)
        {
            var hdrp = (HDRenderPipeline) RenderPipelineManager.currentPipeline;
            hdrp.UpdateShaderVariablesForCamera(cmd, hd);
            context.SetupCameraProperties(hd.camera);

            var originalProj = hd.camera.projectionMatrix;

            var trans = hd.camera.transform;
            var rot = trans.rotation;
            var localRot = trans.localRotation;

            cmd.SetInvertCulling(true);

            for (var i = 0; i < 6; ++i)
            {
                trans.localRotation = localRot * Quaternion.LookRotation(CoreUtils.lookAtList[i], CoreUtils.upVectorList[i]);
                hdrp.SetupGlobalParamsForCubemap(cmd, hd, CubemapSize, out var proj);
                hd.camera.projectionMatrix = proj;

                CoreUtils.SetRenderTarget(cmd, rtColor, rtDepth, depthSlice: i);
                cmd.ClearRenderTarget(true, true, SimulatorManager.Instance.SkySegmentationColor);

                context.ExecuteCommandBuffer(cmd);
                cmd.Clear();

                if (hd.camera.TryGetCullingParameters(out var culling))
                {
                    var cull = context.Cull(ref culling);

                    var sorting = new SortingSettings(hd.camera);
                    var drawing = new DrawingSettings(passId, sorting);
                    var filter = new FilteringSettings(RenderQueueRange.all);

                    context.DrawRenderers(cull, ref drawing, ref filter);
                }
            }

            cmd.SetInvertCulling(false);

            hd.camera.projectionMatrix = originalProj;
            trans.rotation = rot;
        }

        protected override void SensorUpdate()
        {
            SensorCamera.Render();
            maskBuffer.GetData(maskBufferArr);

            DetectedObjects.Clear();

            // 0 is reserved for clear color, 255 for non-agent segmentation
            for (var i = 1; i < 255; ++i)
            {
                if (maskBufferArr[i] == 0)
                    continue;

                var detected = GetDetectedObject(i);
                if (detected != null)
                    DetectedObjects.Add(detected);
            }

            MaxTracked = Math.Max(MaxTracked, DetectedObjects.Count);

            if (Bridge != null && Bridge.Status == Status.Connected)
            {
                var data = new Detected3DObject[DetectedObjects.Count];
                for (var i = 0; i < data.Length; ++i)
                    data[i] = DetectedObjects[i].object3d;

                Publish(new Detected3DObjectData()
                {
                    Name = Name,
                    Frame = Frame,
                    Time = SimulatorManager.Instance.CurrentTime,
                    Sequence = seqId++,
                    Data = data,
                });
            }
        }

        private DetectedObject GetDetectedObject(int segId)
        {
            if (!SimulatorManager.Instance.SegmentationIdMapping.TryGetEntityGameObject(segId, out var go, out var type))
            {
                Debug.LogError($"Entity with ID {segId} is not registered.");
                return null;
            }

            uint id;
            string label;
            Vector3 velocity;
            float angular_speed;  // Angular speed around up axis of objects, in radians/sec

            switch (type)
            {
                case SegmentationIdMapping.SegmentationEntityType.Agent:
                {
                    var controller = go.GetComponent<IAgentController>();
                    var dynamics = go.GetComponent<IVehicleDynamics>();
                    id = controller.GTID;
                    label = "Sedan";
                    velocity = dynamics.Velocity;
                    angular_speed = dynamics.AngularVelocity.y;
                }
                    break;
                case SegmentationIdMapping.SegmentationEntityType.NPC:
                {
                    var npcC = go.GetComponent<NPCController>();
                    id = npcC.GTID;
                    label = npcC.NPCLabel;
                    velocity = npcC.GetVelocity();
                    angular_speed = npcC.GetAngularVelocity().y;
                }
                    break;
                case SegmentationIdMapping.SegmentationEntityType.Pedestrian:
                {
                    var pedC = go.GetComponent<PedestrianController>();
                    id = pedC.GTID;
                    label = "Pedestrian";
                    velocity = pedC.CurrentVelocity;
                    angular_speed = pedC.CurrentAngularVelocity.y;
                }
                    break;
                default:
                {
                    Debug.LogError($"Invalid entity type: {type.ToString()}");
                    return null;
                }
            }

            if (id == Controller.GTID)
                return null;

            var parent = go.transform.gameObject;

            // Linear speed in forward direction of objects, in meters/sec
            float speedX = Vector3.Dot(velocity, parent.transform.forward);
            float speedY = Vector3.Dot(velocity, -parent.transform.right);
            float speedZ = Vector3.Dot(velocity, parent.transform.up);

            // Local position of object in ego local space
            Vector3 relPos = transform.InverseTransformPoint(parent.transform.position);
            // Relative rotation of objects wrt ego frame
            Quaternion relRot = Quaternion.Inverse(transform.rotation) * parent.transform.rotation;

            var mapRotation = MapOrigin.transform.localRotation;
            velocity = Quaternion.Inverse(mapRotation) * velocity;
            var heading = parent.transform.localEulerAngles.y - mapRotation.eulerAngles.y;

            // Center of bounding box
            GpsLocation location = MapOrigin.PositionToGpsLocation(go.transform.position);
            GpsData gps = new GpsData()
            {
                Easting = location.Easting,
                Northing = location.Northing,
                Altitude = location.Altitude,
            };

            if (!SimulatorManager.Instance.SegmentationIdMapping.TryGetEntityLocalBoundingBox(segId, out var bounds))
                return null;

            var obj = new Detected3DObject()
            {
                Id = id,
                Label = label,
                Score = 1.0f,
                Position = relPos,
                Rotation = relRot,
                Scale = bounds.size,
                LinearVelocity = new Vector3(speedX, speedY, speedZ),
                AngularVelocity = new Vector3(0, 0, angular_speed),
                Velocity = velocity,
                Gps = gps,
                Heading = heading,
                TrackingTime = 0f,
            };

            return new DetectedObject
            {
                object3d = obj,
                bounds = bounds,
                transform = parent.transform
            };
        }

        public override void OnVisualize(Visualizer visualizer)
        {
            foreach (var item in DetectedObjects)
            {
                var obj = item.object3d;
                var min = obj.Position - obj.Scale / 2;
                var max = obj.Position + obj.Scale / 2;

                var color = string.Equals(obj.Label, "Pedestrian") ? Color.yellow : Color.green;
                WireframeBoxes.Draw
                (
                    item.transform.localToWorldMatrix,
                    new Vector3(0f, item.bounds.extents.y, 0f),
                    item.bounds.size,
                    color
                );
            }
        }

        public override void OnVisualizeToggle(bool state) {}
    }
}
