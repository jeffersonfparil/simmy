# simmy
Simulate genotype, phenotype, and environmental data for quantitative and population genetics

## Development Plan

- [ ] I/O module
    + [ ] Genomic data struct
    + [ ] Phenomic data struct
    + [ ] Environmental data struct
    + [ ] Read from file
    + [ ] Write to file
- [ ] Genomic data simulation module
    + [ ] Reference genome simulation
    + [ ] LD simulation
    + [ ] Pairwise mating
    + [ ] Panmictic mating
- [ ] Environmental data simulation module
- [ ] Phenomic data simulation module

## Testing

```shell
cd simmy/
pixi shell
mojo src/simmy.mojo
mojo run -I src/ test/array/test_array.mojo
```