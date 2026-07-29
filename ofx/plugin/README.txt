iw3 Stereo
==========

Turns a 2D shot plus a depth pass into stereo 3D, inside DaVinci Resolve.

This is the stereo-synthesis half of iw3 (github.com/nagadomi/nunif). It does
not estimate depth -- you supply that as a clip, from whatever tool you like.


REQUIREMENTS
------------

  * DaVinci Resolve STUDIO. The free version will not load third-party OFX
    plugins at all.
  * An NVIDIA GPU. There is a CPU fallback, but it is roughly twenty times
    slower and is not usable for preview.

  * The CUDA runtime libraries -- cuBLAS and cuDNN. The plugin does not ship
    these: they are about a gigabyte, four times the rest of it. Run

        powershell -ExecutionPolicy Bypass -File fetch-cuda-runtime.ps1

    from an elevated prompt, once, and restart Resolve. It downloads them from
    NVIDIA's own packages and puts them next to the plugin. Nothing is installed
    system-wide and no Python is involved.

    You do NOT need the CUDA Toolkit, and installing it would not be enough on
    its own -- cuDNN is a separate product the Toolkit does not include.

Tested on Resolve 21 and an RTX 5080, where a 1920x800 frame takes about 5 ms.


QUICK START
-----------

1. Go to the FUSION page. This plugin needs two inputs, and Fusion is the only
   place Resolve gives an OFX effect more than one. Dropping it on a timeline
   clip in Edit or Colour will not work -- see LIMITATIONS.

2. Add the node: Effects Library > Open FX > iw3 > iw3 Stereo.

3. Wire your footage into the YELLOW input and your depth pass into the GREEN
   input marked Depth.

You should immediately see a red/cyan anaglyph. If the right half of the image
is flat magenta, the depth input is not connected.


A NOTE ON DATA LEVELS -- LEAVE THEM ALONE
-----------------------------------------

Most video files are tagged "video range", where black is 16 and white is 235
rather than 0 and 255, and Resolve stretches them to fill the range on the way
in. Depth passes written by iw3 are tagged that way too.

It is tempting to think that stretch corrupts a depth map and should be turned
off by setting Clip Attributes > Data Levels to Full. It should not.

iw3 applies exactly the same stretch when it reads its own depth videos. So
leaving Resolve to do it is what makes this plugin agree with iw3. Forcing Data
Levels to Full removes the stretch here but not in iw3, and the two stop
matching.

Measured on the same frame of the same file: iw3's reader gives 0.201 to 1.000
with a mean of 0.669, and what this plugin receives from Resolve is 0.210 to
1.004, mean 0.673. Without the stretch it would be 0.235 to 0.926, mean 0.639 --
clearly a different signal.

So: leave Data Levels at whatever Resolve picked. The Depth Range parameter
exists for the rarer case of a file whose range tag is simply wrong.


PARAMETERS
----------

  Model               Which pipeline to use.

                      row_flow_v3 is the default: a windowed-attention model,
                      generally cleaner around edges. row_flow_v2 is a smaller
                      convolution stack and is still here because neither is
                      more correct than the other.

                      Despite having four times the parameters v3 is also the
                      slightly faster of the two, because it works on a reduced
                      grid: 4.8 ms against 5.4 ms on a 1920x800 frame. Try both
                      on a shot and keep whichever looks better.

                      monobw_inpaint is a different pipeline rather than another
                      network. The other two warp backwards, which means that
                      where an object moves aside there is nothing behind it and
                      the edge gets smeared into the gap. This one warps
                      forwards, works out exactly which pixels ended up with
                      nothing behind them, and fills those with a network. It
                      handles occlusions better and costs roughly eight times
                      as much: about 37 ms a frame at HD against 4 ms, and 4K
                      is render-only.

                      monobw_inpaint_video is the same pipeline with a model
                      that sees twelve frames at once instead of one. The
                      per-frame model invents each fill independently, so the
                      filled regions crawl from frame to frame; this one does
                      not. It asks Resolve for the frames either side of the
                      one being rendered and keeps a window's worth of results,
                      so it costs about twice what monobw_inpaint does rather
                      than twelve times -- roughly 78 ms a frame at HD.

                      mlbw_l2_inpaint is a third pipeline. Like the two
                      above it fills occlusions with a network rather than
                      smearing an edge into them, but it gets there differently:
                      a small network predicts two ways each pixel could have
                      moved, blends them, and predicts where the holes are
                      instead of working them out geometrically. It warps
                      backwards like row_flow rather than forwards like monobw.
                      Worth trying on shots where monobw_inpaint's fills look
                      right but its edges do not, and it costs about the same --
                      they share the same fill network, which is where nearly all
                      the time goes.

                      mlbw_l2_inpaint_video pairs that warp with the same
                      twelve-frame fill monobw_inpaint_video uses, for the same
                      reason: the per-frame model invents each fill on its own,
                      so filled regions crawl. Only the fill sees a window --
                      the warp is per frame either way.

                      All four need the GPU path -- NVIDIA, with Fusion's GPU
                      processing on. On the CPU they decline and pass the
                      source through with a message.

  Mask Inner          Inpaint models only. Grows the hole mask towards the
  Dilation            occluding edge before filling. Worth a try when the depth
                      map's edge sits slightly inside the object's and a rim of
                      the old background survives at the boundary.

  Mask Outer          Inpaint models only. Grows the mask the other way, giving
  Dilation            the network more room to invent into. Worth a try when the
                      fill looks starved next to a large parallax shift.

                      Both are counted against the depth's width, so a setting
                      means the same thing whatever resolution you render at.
                      Both do nothing when another Model is selected.

  Inpaint Max        Caps the width the inpaint network runs at: Full, 1920,
  Width              1280, 960 or 720. Full is the frame's own width, which on
                     a 4K timeline is not the same as picking 1920.

                      Defaults to 1280, so that trying an inpaint model does not
                      immediately ask your card for nine gigabytes. If yours has
                      the memory, Full is the better picture.

                      This is the setting to reach for if you run out of VRAM.
                      The network's memory scales with area: at HD the temporal
                      model wants about 9 GB, at 1280 wide about 4.5, at 960
                      about 3.4. It gets faster in the same proportion.

                      It does not shrink your output. The warp and the hole
                      detection stay at full resolution; only the network runs
                      small, and what it invents is blended back into the
                      full-resolution frame. Everything outside a hole keeps its
                      own detail. Since the filled pixels are invented anyway,
                      this is a cheap place to save.

Stereo

  Divergence          Strength of the 3D effect, as a percentage of image
                      width. Default 2.0. Up to about 2 is comfortable; higher
                      is more dramatic and more prone to artefacts around
                      edges. Both eyes are synthesised, each carrying half.

  Convergence         Which depth sits on the screen plane. Default 0.5.
                      0 pushes the whole scene behind the screen, 1 brings it
                      all in front. Behind is easier on the eyes.

  Preserve Screen     Fades the parallax to nothing at the left and right
  Border              edges. Helps when objects run off the side of frame.

Depth

  Depth Is Inverted   Turn on if your depth pass has near = black. Tools
                      disagree about this and there is no way to detect it.
                      If the 3D looks inside-out, this is the switch.

  Foreground Scale    Reshapes the depth to push the foreground forward
                      (positive) or flatten it (negative). 0 is off.

  Mapper Type         Which family Foreground Scale draws from. Only matters
                      when Foreground Scale is not 0.

  Stereo Width        Width the depth is reduced to before warping. Leave at 0
                      unless you have a reason. See below.

  Depth Range         Leave on "As delivered". Resolve's range handling
                      already matches what iw3 does. "Undo video expansion" is
                      for a file whose range tag is wrong, and will make a
                      correctly tagged one worse.

Output

  Anaglyph            Red/cyan, using Dubois coefficients. The default,
                      because it makes the effect visible at a glance.
  Left eye            The synthesised left view on its own.
  Right eye           The synthesised right view on its own.
  Half SBS            Both eyes squeezed side by side into one frame.
  Depth (debug)       What the model actually sees, after all depth
                      processing. Useful for checking Stereo Width and
                      Depth Is Inverted.

For a proper stereo pair, use two copies of the node -- one set to Left eye,
one to Right eye -- and combine them however your delivery needs.


FULL SIDE BY SIDE
-----------------

There is no Full SBS option, and there cannot be one.

An OFX effect cannot output an image larger than its input unless the host
supports multiple resolutions, and Resolve's OFX host reports that it does not
(kOfxImageEffectPropSupportsMultiResolution = 0). It was implemented and tried:
the plugin asked for a double-width output and Resolve declined, which the
specification entitles it to do.

Fusion has stereo tools of its own with no such limit, so build it there:

  1. Add two iw3 Stereo nodes fed from the same Source and Depth. Set one to
     Output: Left eye and the other to Output: Right eye.

  2. Add a Combiner. Connect the left-eye node to Image1 and the right-eye
     node to Image2. This makes a single stereo image.

  3. Add an Anaglyph node after it and switch on Horizontal Stack. Despite the
     name, that is what writes the two views side by side at full width.

  4. Feed it to MediaOut.

Each eye stays at its native resolution, which is what Full SBS is for. The
Half SBS option squeezes both into one frame instead, and is the right choice
when your delivery expects that.

Every setting on the two iw3 Stereo nodes must match except Output, or the eyes
will not correspond. Rather than change both by hand each time, it is worth
adding a controller node: one parked off to the side, connected to nothing, with
user controls for divergence, convergence and the rest, and both iw3 nodes
linked to it by simple expressions such as

    iw3Control.Divergence

Then one panel drives both, and the two cannot drift apart.

An sMerge (Shape category) is a good host for it. It works on vector shapes
rather than images, so unconnected it allocates nothing and brings almost no
controls of its own to crowd out yours.


ABOUT STEREO WIDTH
------------------

Resolve always hands the plugin depth at the composition's resolution, whatever
size the depth file really is. That matters: the warp takes its scale from the
depth's width, and the model was trained on depth at the size a depth estimator
produces. Give it a full-resolution depth map and the picture develops fine
vertical stripes.

So Stereo Width = 0 does not mean "leave it alone". It means "reduce the depth
to the size a depth model would have produced", which for a 1920x800 comp is
about 938x392.

Set it explicitly only if you want to override that -- 1280 or 640 are sensible
values. Use the Depth (debug) output to see the result.


LIMITATIONS
-----------

  * Fusion page only. On the Edit and Colour pages Resolve gives an OFX effect
    a single input, so there is nowhere for the depth to arrive. The node will
    appear there but passes the image through untouched.

  * NVIDIA only for GPU speed. The plugin will run on the CPU if it must, and
    warns you in the node when it does, but expect roughly 250 ms a frame
    rather than 5.

  * Full SBS is not offered, only Half SBS. An OFX plugin cannot enlarge its
    output frame in Resolve.


IF SOMETHING IS WRONG
---------------------

The plugin writes a log to:

    %LOCALAPPDATA%\iw3probe\probe.log

Paste that into a bug report. It records which GPU path started, how long
start-up took, and any error from the inference runtime.

Common cases:

  Image unchanged, no error
      Depth input not connected, or you are on the Edit/Colour page rather
      than Fusion.

  Warning about running on the CPU
      The CUDA runtime did not start. The log says why. If it mentions
      cublasLt64_13.dll, run fetch-cuda-runtime.ps1 -- see REQUIREMENTS.
      Otherwise check your NVIDIA driver.

  Anaglyph looks right but the inpaint models produce nothing
      If the log says "cuDNN is unavailable or disabled", cuBLAS is present but
      cuDNN is not. That is the same fix: fetch-cuda-runtime.ps1. The two
      libraries fail differently -- cuBLAS stops the GPU path from starting at
      all, cuDNN lets it start and then fails every convolution.

  3D looks inside-out
      Turn on Depth Is Inverted.

  Fine vertical stripes over the picture
      Stereo Width has been set too high. Return it to 0.


LICENCE
-------

MIT. Derived from nunif/iw3 by nagadomi, also MIT, and shipped with the
author's permission. See LICENSE and NOTICE in the source repository.
