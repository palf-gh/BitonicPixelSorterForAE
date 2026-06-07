using System;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
#if BPS_URP_17
using UnityEngine.Rendering.RenderGraphModule;
#endif

namespace Ruccho.Utilities
{
    public sealed class BitonicPixelSortingFeature : ScriptableRendererFeature
    {
        [SerializeField] private ComputeShader shader;
        [SerializeField] private bool direction = true;
        [SerializeField] private bool ascending = true;
        [SerializeField] [Range(0f, 1f)] private float thresholdMin = 0.4f;
        [SerializeField] [Range(0f, 1f)] private float thresholdMax = 0.6f;

        public RenderPassEvent passEvent = RenderPassEvent.AfterRenderingTransparents;
        private BitonicPixelSortingPass _pass;

        public override void Create()
        {
            _pass = new BitonicPixelSortingPass();
        }

        public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData)
        {
            _pass.renderPassEvent = passEvent;
            _pass.Shader = shader;
            _pass.Direction = direction;
            _pass.Ascending = ascending;
            _pass.ThresholdMin = thresholdMin;
            _pass.ThresholdMax = thresholdMax;

            renderer.EnqueuePass(_pass);
        }

        private class BitonicPixelSortingPass : ScriptableRenderPass
        {
            private const string Tag = nameof(BitonicPixelSortingPass);
            private const string KeywordSize4096 = "BPS_SIZE_4096";

            private static readonly int PropDirection = UnityEngine.Shader.PropertyToID("direction");
            private static readonly int PropOrdering = UnityEngine.Shader.PropertyToID("ordering");
            private static readonly int PropSortTex = UnityEngine.Shader.PropertyToID("sortTex");
            private static readonly int PropSrcTex = UnityEngine.Shader.PropertyToID("srcTex");
            private static readonly int PropThresholdMax = UnityEngine.Shader.PropertyToID("thresholdMax");
            private static readonly int PropThresholdMin = UnityEngine.Shader.PropertyToID("thresholdMin");
            public bool Ascending = true;
            public bool Direction = true;

            public ComputeShader Shader;
            public float ThresholdMax = 0.6f;
            public float ThresholdMin = 0.4f;

#if BPS_URP_17
            [Obsolete(
                "This rendering path is for compatibility mode only (when Render Graph is disabled). Use Render Graph API instead.",
                false)]
#endif
            public override void Execute(ScriptableRenderContext context, ref RenderingData renderingData)
            {
                var cmd = CommandBufferPool.Get(Tag);

                var sortPassIndex = Shader.FindKernel("SortPass");

                var desc = renderingData.cameraData.cameraTargetDescriptor;


                var width = desc.width;
                var height = desc.height;
                var size = Direction ? width : height;
                var lines = Direction ? height : width;

                if (size > 4096)
                {
                    Debug.LogError("[BitonicPixelSorter] Size of source texture must be 4096 or smaller.");
                    return;
                }

                if (size > 2048) Shader.EnableKeyword(KeywordSize4096);
                else Shader.DisableKeyword(KeywordSize4096);

                var renderer = renderingData.cameraData.renderer;
                var src = renderer.cameraColorTargetHandle;

                cmd.GetTemporaryRT(PropSortTex, width, height, 0, FilterMode.Point, RenderTextureFormat.ARGB32,
                    RenderTextureReadWrite.Default, 1, true);

                cmd.SetComputeIntParam(Shader, PropDirection, Direction ? 1 : 0);
                cmd.SetComputeFloatParam(Shader, PropThresholdMin, ThresholdMin);
                cmd.SetComputeFloatParam(Shader, PropThresholdMax, ThresholdMax);

                cmd.SetComputeTextureParam(Shader, sortPassIndex, PropSrcTex, src);
                cmd.SetComputeTextureParam(Shader, sortPassIndex, PropSortTex, new RenderTargetIdentifier(PropSortTex));

                cmd.SetComputeIntParam(Shader, PropOrdering, Ascending ? 1 : 0);

                cmd.DispatchCompute(Shader, sortPassIndex, lines, 1, 1);

                cmd.Blit(new RenderTargetIdentifier(PropSortTex), src);

                context.ExecuteCommandBuffer(cmd);
                cmd.ReleaseTemporaryRT(PropSortTex);
                CommandBufferPool.Release(cmd);
            }

#if BPS_URP_17

            public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
            {
                var resourcesData = frameData.Get<UniversalResourceData>();
                var cameraData = frameData.Get<UniversalCameraData>();

                var desc = cameraData.cameraTargetDescriptor;

                var width = desc.width;
                var height = desc.height;
                var size = Direction ? width : height;
                var lines = Direction ? height : width;

                if (size > 4096)
                {
                    Debug.LogError("[BitonicPixelSorter] Size of source texture must be 4096 or smaller.");
                    return;
                }

                var sortTex = UniversalRenderer.CreateRenderGraphTexture(renderGraph,
                    new RenderTextureDescriptor(width, height, RenderTextureFormat.ARGB32)
                    {
                        enableRandomWrite = true
                    },
                    "_BpsSortTex", false);

                var cameraTarget = resourcesData.activeColorTexture;

                var source = cameraTarget;

                using (var builder =
                       renderGraph.AddComputePass<SortPassData>("BitonicPixelSorter SortPass", out var passData,
                           profilingSampler))
                {
                    builder.UseTexture(sortTex, AccessFlags.WriteAll);
                    builder.UseTexture(source);

                    passData.Shader = Shader;
                    passData.Ascending = Ascending;
                    passData.Direction = Direction;
                    passData.Lines = lines;
                    passData.Use4096 = size > 2048;
                    passData.ThresholdMax = ThresholdMax;
                    passData.ThresholdMin = ThresholdMin;
                    passData.SortTex = sortTex;
                    passData.SrcTex = source;

                    builder.SetRenderFunc<SortPassData>(static (passData, context) =>
                    {
                        var cmd = context.cmd;
                        var shader = passData.Shader;
                        var direction = passData.Direction;
                        var ascending = passData.Ascending;
                        var lines = passData.Lines;
                        var thresholdMax = passData.ThresholdMax;
                        var thresholdMin = passData.ThresholdMin;

                        if (passData.Use4096) shader.EnableKeyword(KeywordSize4096);
                        else shader.DisableKeyword(KeywordSize4096);

                        var sortPassIndex = shader.FindKernel("SortPass");

                        cmd.SetComputeIntParam(shader, PropDirection, direction ? 1 : 0);
                        cmd.SetComputeFloatParam(shader, PropThresholdMin, thresholdMin);
                        cmd.SetComputeFloatParam(shader, PropThresholdMax, thresholdMax);

                        cmd.SetComputeTextureParam(shader, sortPassIndex, PropSrcTex, passData.SrcTex);
                        cmd.SetComputeTextureParam(shader, sortPassIndex, PropSortTex, passData.SortTex);

                        cmd.SetComputeIntParam(shader, PropOrdering, ascending ? 1 : 0);

                        cmd.DispatchCompute(shader, sortPassIndex, lines, 1, 1);
                    });
                }

                using (var builder =
                       renderGraph.AddRasterRenderPass<BlitPassData>("BitonicPixelSorter Destination Blit",
                           out var passData,
                           profilingSampler))
                {
                    builder.UseTexture(sortTex);
                    builder.SetRenderAttachment(source, 0, AccessFlags.WriteAll);

                    passData.Source = sortTex;

                    builder.SetRenderFunc<BlitPassData>(static (passData, context) =>
                    {
                        Blitter.BlitTexture(context.cmd, passData.Source, new Vector4(1f, 1f, 0f, 0f), 0f, false);
                    });
                }
            }

            private class SortPassData
            {
                public bool Ascending;
                public bool Direction;
                public int Lines;
                public bool Use4096;

                public TextureHandle SrcTex;
                public ComputeShader Shader;
                public TextureHandle SortTex;
                public float ThresholdMax;
                public float ThresholdMin;
            }

            private class BlitPassData
            {
                public TextureHandle Source;
            }

#endif
        }
    }
}