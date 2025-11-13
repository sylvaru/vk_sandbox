// tonemapping.glsl
#ifndef TONEMAPPING_GLSL
#define TONEMAPPING_GLSL

// ACES Filmic Tone Mapping (Narkowicz 2015)
vec3 ACESFilm(vec3 x)
{
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

// gamma correction
vec3 toSRGB(vec3 x) {
    return pow(x, vec3(1.0 / 2.2));
}
vec3 linearToSrgb(vec3 c) { return pow(c, vec3(1.0 / 2.2)); }
vec3 srgbToLinear(vec3 c) { return pow(c, vec3(2.2)); }
#endif
