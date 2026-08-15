import { Object3D } from "../core/Object3D";
import { BufferGeometry } from "../geometries/BufferGeometry";
import { Material } from "../materials/Material";

/**
 * A drawable object: a `geometry` (vertex buffers) and a `material` (the
 * shader + render state). Like three.js's `Mesh`.
 */
export class Mesh extends Object3D {
  readonly isMesh = true;
  geometry: BufferGeometry;
  material: Material;
  castShadow = false;
  receiveShadow = false;

  constructor(geometry: BufferGeometry = new BufferGeometry(), material: Material = new Material()) {
    super();
    this.geometry = geometry;
    this.material = material;
  }
}
