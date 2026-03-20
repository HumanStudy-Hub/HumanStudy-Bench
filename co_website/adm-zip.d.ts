declare module "adm-zip" {
  interface ZipEntry {
    isDirectory: boolean;
    entryName: string;
    getData(): Buffer;
  }

  class AdmZip {
    constructor(buffer?: Buffer);
    getEntries(): ZipEntry[];
  }

  export default AdmZip;
}
