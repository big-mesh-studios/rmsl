import { type UniformNode } from "../../../rmsl";
import type { Scene } from "../../scenes/Scene";
import { AmbientLight } from "../../lights/AmbientLight";
import { DirectionalLight } from "../../lights/DirectionalLight";
import { PointLight } from "../../lights/PointLight";
import type { Builder } from "./Builder";

export interface DirectionalLightState {
  color: UniformNode<"vec3">;
  direction: UniformNode<"vec3">;
}

export interface PointLightState {
  color: UniformNode<"vec3">;
  position: UniformNode<"vec3">;
  distance: UniformNode<"float">;
  decay: UniformNode<"float">;
}

export interface LightUniforms {
  ambient: UniformNode<"vec3">;
  directionals: DirectionalLightState[];
  points: PointLightState[];
}

/**
 * Register the light uniforms for the lights present in `scene`. Ambient
 * lights fold into one `ambientColor`; each directional and point light gets
 * its own color/direction/position uniforms, all material-scoped and read live
 * from the light objects on every upload.
 */
export function collectLights(b: Builder, scene: Scene): LightUniforms {
  const ambientLights: AmbientLight[] = [];
  const directionalLights: DirectionalLight[] = [];
  const pointLights: PointLight[] = [];

  scene.traverseVisible((object) => {
    if (object instanceof AmbientLight) ambientLights.push(object);
    else if (object instanceof DirectionalLight) directionalLights.push(object);
    else if (object instanceof PointLight) pointLights.push(object);
  });

  const ambient = b.materialUniform("ambientColor", "vec3", () => {
    let r = 0, g = 0, bl = 0;
    for (const light of ambientLights) {
      r += light.color.r * light.intensity;
      g += light.color.g * light.intensity;
      bl += light.color.b * light.intensity;
    }
    return [r, g, bl];
  });

  const directionals = directionalLights.map((light, i) => ({
    color: b.materialUniform(`directionalColor${i}`, "vec3", () => {
      const [r, g, bl] = light.color.toArray();
      return [r * light.intensity, g * light.intensity, bl * light.intensity];
    }),
    direction: b.materialUniform(`directionalDirection${i}`, "vec3", () =>
      light.getWorldPosition()
        .sub(light.target.getWorldPosition())
        .normalize()
        .toArray(),
    ),
  }));

  const points = pointLights.map((light, i) => ({
    color: b.materialUniform(`pointColor${i}`, "vec3", () => {
      const [r, g, bl] = light.color.toArray();
      return [r * light.intensity, g * light.intensity, bl * light.intensity];
    }),
    position: b.materialUniform(`pointPosition${i}`, "vec3", () =>
      light.getWorldPosition().toArray(),
    ),
    distance: b.materialUniform(`pointDistance${i}`, "float", () => light.distance),
    decay: b.materialUniform(`pointDecay${i}`, "float", () => light.decay),
  }));

  return { ambient, directionals, points };
}
