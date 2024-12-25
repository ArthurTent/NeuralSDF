import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
import re
from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import skimage
import matplotlib.pyplot as plt

from math import sqrt

import time

from mesh_to_sdf import get_surface_point_cloud
from mesh_to_sdf.utils import sample_uniform_points_in_unit_sphere

import trimesh
import pyrender

CUDA = False


def get_mgrid(sidelen, dim=2):
    """Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int"""
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        is_first=False,
        omega_0=30,
        has_skip=False,
        skip_idx=1,
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.has_skip = has_skip
        self.skip_idx = skip_idx

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(
                    -1.0 / self.in_features, 1.0 / self.in_features
                )
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0,
                )

    def forward(self, input):
        intermediate = torch.sin(self.omega_0 * self.linear(input))
        if self.has_skip:
            intermediate = intermediate / self.skip_idx + input
        return intermediate

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        hidden_layers,
        out_features,
        outermost_linear=False,
        omega=30,
        first_linear=False,
    ):
        super().__init__()
        self.omega = omega
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_layers = hidden_layers
        self.hidden_features = hidden_features
        self.first_linear = first_linear
        self.net = []
        if first_linear:
            linear = nn.Linear(in_features, hidden_features)
            with torch.no_grad():
                linear.weight.uniform_(
                    -1.0 / self.in_features / omega, 1.0 / self.in_features / omega
                )
            self.net.append(linear)
        else:
            self.net.append(
                SineLayer(in_features, hidden_features, is_first=True, omega_0=omega)
            )

        for i in range(hidden_layers):
            self.net.append(
                SineLayer(
                    hidden_features,
                    hidden_features,
                    is_first=False,
                    omega_0=omega,
                    has_skip=True,
                    skip_idx=sqrt(i + 1),
                )
            )

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(
                    -np.sqrt(6 / hidden_features) / omega,
                    np.sqrt(6 / hidden_features) / omega,
                )

            self.net.append(final_linear)
        else:
            self.net.append(
                SineLayer(hidden_features, out_features, is_first=False, omega_0=omega)
            )

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = (
            coords.clone().detach().requires_grad_(True)
        )  # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords

    def forward_with_activations(self, coords, retain_grad=False):
        """Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!"""
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations["input"] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)

                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()

                activations[
                    "_".join((str(layer.__class__), "%d" % activation_count))
                ] = intermed
                activation_count += 1
            else:
                x = layer(x)

                if retain_grad:
                    x.retain_grad()

            activations["_".join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations


class SDFFitting(Dataset):
    def __init__(self, filename, samples):
        super().__init__()
        mesh = trimesh.load(filename)
        # mesh, number_of_points = 500000, surface_point_method='scan', sign_method='normal', scan_count=100, scan_resolution=400, sample_point_count=10000000, normal_sample_count=11, min_size=0
        surface_point_cloud = get_surface_point_cloud(
            mesh, surface_point_method="sample"
        )

        self.coords, self.samples = surface_point_cloud.sample_sdf_near_surface(
            samples // 2, use_scans=False, sign_method="normal"
        )
        unit_sphere_points = sample_uniform_points_in_unit_sphere(samples // 2)
        samples = surface_point_cloud.get_sdf_in_batches(
            unit_sphere_points, use_depth_buffer=False
        )
        self.coords = np.concatenate([self.coords, unit_sphere_points]).astype(
            np.float32
        )
        self.samples = np.concatenate([self.samples, samples]).astype(np.float32)

        # colors = np.zeros(self.coords.shape)
        # colors[self.samples < 0, 2] = 1
        # colors[self.samples > 0, 0] = 1
        # cloud = pyrender.Mesh.from_points(self.coords, colors=colors)
        # scene = pyrender.Scene()
        # scene.add(cloud)
        # viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)

        self.samples = torch.from_numpy(self.samples)[:, None]
        self.coords = torch.from_numpy(self.coords)
        print(self.coords.shape, self.samples.shape)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0:
            raise IndexError

        return self.coords, self.samples


# HERE


def dump_data(dat):
    dat = dat.cpu().detach().numpy()
    return dat


def print_vec4(ws, precision):
    # precision
    # vec = "vec4(" + ",".join(["{0:.2f}".format(w) for w in ws]) + ")"
    # vec = "vec4(" + ",".join(["{0:.3f}".format(w) for w in ws]) + ")"
    format_str = f"{{0:.{precision}f}}"
    vec = "vec4(" + ",".join([format_str.format(w) for w in ws]) + ")"
    vec = re.sub(r"\b0\.", ".", vec)
    return vec


def print_mat4(ws, precision):
    # precision
    # mat = "mat4(" + ",".join(["{0:.2f}".format(w) for w in np.transpose(ws).flatten()]) + ")"
    # mat = "mat4(" + ",".join(["{0:.3f}".format(w) for w in np.transpose(ws).flatten()]) + ")"
    format_str = f"{{0:.{precision}f}}"
    mat = (
        "mat4("
        + ",".join([format_str.format(w) for w in np.transpose(ws).flatten()])
        + ")"
    )

    mat = re.sub(r"\b0\.", ".", mat)
    return mat


def serialize_to_shadertoy(siren, varname, precision):
    result = ""
    # first layer
    omega = siren.omega
    chunks = int(siren.hidden_features / 4)
    lin = siren.net[0] if siren.first_linear else siren.net[0].linear
    in_w = dump_data(lin.weight)
    in_bias = dump_data(lin.bias)
    om = 1 if siren.first_linear else omega
    for row in range(chunks):
        if siren.first_linear:
            line = "vec4 %s0_%d=(" % (varname, row)
        else:
            line = "vec4 %s0_%d=sin(" % (varname, row)

        for ft in range(siren.in_features):
            feature = x_vec = in_w[row * 4 : (row + 1) * 4, ft] * om
            line += (
                ("p.%s*" % ["y", "z", "x"][ft]) + print_vec4(feature, precision) + "+"
            )
        bias = in_bias[row * 4 : (row + 1) * 4] * om
        line += print_vec4(bias, precision) + ");"
        print(line)
        result += line

    # hidden layers
    for layer in range(siren.hidden_layers):
        layer_w = dump_data(siren.net[layer + 1].linear.weight)
        layer_bias = dump_data(siren.net[layer + 1].linear.bias)
        for row in range(chunks):
            line = ("vec4 %s%d_%d" % (varname, layer + 1, row)) + "=sin("
            for col in range(chunks):
                mat = layer_w[row * 4 : (row + 1) * 4, col * 4 : (col + 1) * 4] * omega
                line += (
                    print_mat4(mat, precision)
                    + ("*%s%d_%d" % (varname, layer, col))
                    + "+\n    "
                )
            bias = layer_bias[row * 4 : (row + 1) * 4] * omega
            line += print_vec4(bias, precision) + ")/%0.1f+%s%d_%d;" % (
                sqrt(layer + 1),
                varname,
                layer,
                row,
            )
            print(line)
            result += line

    # output layer
    out_w = dump_data(siren.net[-1].weight)
    out_bias = dump_data(siren.net[-1].bias)
    for outf in range(siren.out_features):
        line = "return "
        # result += "return "
        for row in range(chunks):
            vec = out_w[outf, row * 4 : (row + 1) * 4]
            line += (
                ("dot(%s%d_%d," % (varname, siren.hidden_layers, row))
                + print_vec4(vec, precision)
                + ")+\n    "
            )
            # result += ("dot(%s%d_%d,"%(varname, siren.hidden_layers, row)) + print_vec4(vec, precision) + ")+\n    "
        print(line + "{:0.3f}".format(out_bias[outf]) + ";")
        result += line + "{:0.3f}".format(out_bias[outf]) + ";"

    return result


def serialize_to_shaderamp(siren, varname, precision):
    shaderamp_header = """
// https://www.shadertoy.com/view/7sSSDd
// Modified by ArthurTent
// Created by Quasimondo
// Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.
// https://creativecommons.org/licenses/by-nc-sa/3.0/

uniform float iAmplifiedTime;
uniform float iTime;
uniform sampler2D iAudioData;
uniform vec2 iResolution;
uniform vec2 iMouse;
varying vec2 vUv;


#define FFT(a) pow(texelFetch(iAudioData, ivec2(a, 0), 0).x, 5.)
float snd = 0.;
const float PI = 3.1415926;

// MIT Licensed hash From Dave_Hoskins (https://www.shadertoy.com/view/4djSRW)
vec3 hash33(vec3 p)
{
    p = fract(p * vec3(443.8975,397.2973, 491.1871));
    p += dot(p.zxy, p.yxz+19.27);
    return fract(vec3(p.x * p.y, p.z*p.x, p.y*p.z));
}

vec3 stars(in vec3 p)
{
    vec3 c = vec3(0.);
    float res = iResolution.x*0.8;
    
	for (float i=0.;i<4.;i++)
    {
        vec3 q = fract(p*(.15*res))-0.5;
        //q*= snd/10.;
        vec3 id = floor(p*(.15*res));
        vec2 rn = hash33(id).xy;
        float c2 = 1.-smoothstep(0.,.6,length(q));
        c2 *= step(rn.x,.0005+i*i*0.001);
        c += c2*(mix(vec3(1.0,0.49,0.1),vec3(0.75,0.9,1.),rn.y)*0.25+0.75);
        p *= 1.4;
    }
    return c*c*.65;
}
void camera(vec2 fragCoord, out vec3 ro, out vec3 rd, out mat3 t)
{
    float a = 1.0/max(iResolution.x, iResolution.y);
    //rd = normalize(vec3((fragCoord - iResolution.xy*0.5)*a, 0.5));
    rd = normalize(vec3(fragCoord, 1.0));

    ro = vec3(0.0, 0.0, -15.);

    //float ff = min(1.0, step(0.001, iMouse.x) + step(0.001, iMouse.y));
    float ff = min(1.0, step(0.001, iMouse.x) + step(0.001, iMouse.y))+sin(iTime/20.);
    vec2 m = PI*ff + vec2(((iMouse.xy + 0.1) / iResolution.xy) * (PI*2.0));
    //m.y = -m.y;
    m.y = sin(m.y*0.5)*0.3 + 0.5;

    //vec2 sm = sin(m)*sin(iTime), cm = cos(m)*(1.+sin(iTime));
    vec2 sm = sin(m)*(1.+sin(iTime/10.)/2.), cm = cos(m);
    mat3 rotX = mat3(1.0, 0.0, 0.0, 0.0, cm.y, sm.y, 0.0, -sm.y, cm.y);
    mat3 rotY = mat3(cm.x, 0.0, -sm.x, 0.0, 1.0, 0.0, sm.x, 0.0, cm.x);

    t = rotY * rotX;

    ro = t * ro;
    rd = t * rd;

    rd = normalize(rd);
}
vec3 palette(float t) {
    if(t <1.)t+=1.;
    vec3 a = vec3(0.5);
    vec3 b = vec3(0.5);
    vec3 c = vec3(1.);
    vec3 d = vec3(0.563,0.416,0.457 + .2);
    
    return a + b*cos( 6.28 * c * (t+d)); // A + B * cos ( 2pi * (Cx + D) )
}


//Parts of this shader code are based on the work of Blackle Mori / https://www.shadertoy.com/view/wtVyWK
//Siren model trained and customized by Mario Klingemann / @Quasimondo

float scene(vec3 p) {
    //sdf is undefined outside the unit sphere, uncomment to witness the abominations
    if (length(p) > 1.) {
        return length(p)-.8;
    }
    p/=1.+snd;
    //neural networks can be really compact... when they want to be
    """

    shaderamp_footer = """
} 

vec3 norm(vec3 p) {
    mat3 k = mat3(p,p,p)-mat3(0.001);
    return normalize(scene(p) - vec3(scene(k[0]),scene(k[1]),scene(k[2])));
}

vec3 erot(vec3 p, vec3 ax, float ro) {
    return mix(dot(p,ax)*ax,p,cos(ro))+sin(ro)*cross(ax,p);
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    int max_freq = 100;
    for(int i=1; i < max_freq; i++){
        snd +=FFT(i)*float(i);
    }
    snd /=float(max_freq*20);
    

    vec2 uv = (fragCoord-0.5*iResolution.xy)/iResolution.y;
    vec2 mouse = (iMouse.xy-0.5*iResolution.xy)/iResolution.y;
    vec3 cam = normalize(vec3(1.5,uv));
    vec2 cam_uv = (fragCoord-.5*iResolution.xy)/iResolution.y;
    
    //camera + rd for stars
    vec3 ro = vec3(0.0);//rd = vec3( 0.0 );
	vec3 rd = normalize(vec3(cam_uv,-1.5));
    mat3 t3 = mat3(1.0);
	camera(uv, ro, rd, t3);
    
    vec3 init = vec3(-3.,0,0);
    //vec3 rd = normalize(vec3(cam_uv,-1.5));
    rd.x+=sin(iTime/1000.)*2.;
    
    float yrot = 0.5;
    float zrot = -iTime*1.42;
    /*
	if (iMouse.z > 0.) {
        yrot += -4.*mouse.y;
        zrot = 4.*mouse.x;
    }
	*/
    cam = erot(cam, vec3(0,1,0), yrot);
    init = erot(init, vec3(0,1,0), yrot);
    cam = erot(cam, vec3(0,0,1), zrot);
    init = erot(init, vec3(0,0,1), zrot);
    
    vec3 p = init;
    bool hit = false;
    for (int i = 0; i < 150 && !hit; i++) {
        float dist = scene(p);
        hit = dist*dist < 1e-6;
        p+=dist*cam;
        if (distance(p,init)>5.) break;
    }
    vec3 n = norm(p);
    vec3 r = reflect(cam,n);
    //don't ask how I stumbled on this texture
    vec3 nz = p - erot(p, vec3(1), 2.) + erot(p, vec3(1), 4.);
    float spec = length(sin(r*3.5+sin(nz*120.)*.15)*.4+.6)/sqrt(3.);
    spec *= smoothstep(-.3,.2,scene(p+r*.2));
    vec3 col = vec3(.1,.1,.12)*spec + pow(spec,8.);
    float bgdot = length(sin(cam*8.)*.4+.6)/2.;
    //vec3 bg = vec3(.1,.1,.11) * bgdot + pow(bgdot, 10.);
    vec3 bg = stars(rd)*(1.+30.*snd);
    //fragColor.xyz = hit ? col *.75+palette(snd*2.+sin(iTime/10.)): bg;
    fragColor.xyz = hit ? col *palette(snd*2.+sin(iTime/10.))*2.: bg;
    fragColor = smoothstep(-.02,1.05,sqrt(fragColor)) * (1.- dot(uv,uv)*.5);
}

void main() {
	vec2 fragCoord = vUv * iResolution;
	mainImage(gl_FragColor, fragCoord);
}

    """
    result = serialize_to_shadertoy(siren, "f", precision)
    """fd = os.open("shaderamp.frag", os.O_RDWR|os.O_CREAT)
    os.write(fd, shaderamp_header+result+shaderamp_footer)
    os.close(fd)"""
    with open("shaderamp.frag", "w") as f:
        f.write(shaderamp_header)
        f.write(result)
        f.write(shaderamp_footer)


def train_siren(dataloader, hidden_features, hidden_layers, omega, cuda=False):
    print("train_siren")
    model_input, ground_truth = next(iter(dataloader))
    if cuda:
        model_input, ground_truth = model_input.cuda(), ground_truth.cuda()
    else:
        model_input, ground_truth = model_input, ground_truth

    img_curr = Siren(
        in_features=3,
        out_features=1,
        hidden_features=hidden_features,
        hidden_layers=hidden_layers,
        outermost_linear=True,
        omega=omega,
        first_linear=False,
    )
    if cuda:
        img_curr.cuda()
    else:
        img_curr
    # optim = torch.optim.Adagrad(params=img_curr.parameters())
    # optim = torch.optim.Adam(lr=1e-3, params=img_curr.parameters())
    optim = torch.optim.Adam(lr=1e-4, params=img_curr.parameters(), weight_decay=0.01)
    perm = torch.randperm(model_input.size(1))

    total_steps = 20000
    update = int(total_steps / 50)
    batch_size = 256 * 256
    for step in range(total_steps):
        if step == 500:
            optim.param_groups[0]["weight_decay"] = 0.0
        idx = step % int(model_input.size(1) / batch_size)
        model_in = model_input[:, perm[batch_size * idx : batch_size * (idx + 1)], :]
        truth = ground_truth[:, perm[batch_size * idx : batch_size * (idx + 1)], :]
        model_output, coords = img_curr(model_in)

        loss = (model_output - truth) ** 2
        loss = loss.mean()

        optim.zero_grad()
        loss.backward()
        optim.step()

        if (step % update) == update - 1:
            perm = torch.randperm(model_input.size(1))
            print("Step %d, Current loss %0.6f" % (step, loss))

    return img_curr


def load(file):
    sdf = SDFFitting(file, 256 * 256 * 4)
    sdfloader = DataLoader(sdf, batch_size=1, pin_memory=False, num_workers=0)
    return sdfloader


"""
sdf = SDFFitting("shaderamp_text.obj", 256*256*4)
#sdf = SDFFitting("voodoo_scaled2.obj", 256*256*4)
#sdf = SDFFitting("IPOD_scaled.obj", 256*256*4) # broken 
#sdf = SDFFitting("keyboard_scaled.obj", 256*256*4) # broken
#sdf = SDFFitting("me_pcd.obj", 256*256*4)
#sdf = SDFFitting("pjanic.obj", 256*256*4)
#sdf = SDFFitting("FinalBaseMeshScaled.obj", 256*256*4)
#sdf = SDFFitting("Corona/Corona_scaled.obj", 256*256*4)
#sdf = SDFFitting("Corona/Corona.obj", 256*256*4)
#sdf = SDFFitting("bunny2.obj", 256*256*4)
#sdf = SDFFitting("FinalBaseMesh.obj", 256*256*4)
#sdf = SDFFitting("11665_Airplane_v1_l3.obj", 256*256*4)
sdfloader = DataLoader(sdf, batch_size=1, pin_memory=False, num_workers=0)

#sdf_siren = train_siren(sdfloader, 64, 16, 15, True)
sdf_siren = train_siren(sdfloader, 16, 4, 15, True)
#sdf_siren = train_siren(sdfloader, 32, 8, 30)

serialize_to_shadertoy(sdf_siren, "f", 2)
"""
