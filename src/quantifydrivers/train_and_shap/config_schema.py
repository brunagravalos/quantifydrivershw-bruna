from pydantic import BaseModel, Field, validator
from typing import List,Literal
from datetime import date
from omegaconf import DictConfig, OmegaConf

class PathsConfig(BaseModel):
    base_folder: str
    model_dir: str
    results_dir: str

    file_spei: str
    file_spi: str

    hyperparameters_dir: str

class SiteHyperparameters(BaseModel):
    lr: float = Field(
        ...,
        ge=1e-5,
        le=1e-3,
        description="Learning rate (Optuna range: [1e-5, 1e-3], log-uniform)"
    )

    w_decay: float = Field(
        ...,
        ge=1e-5,
        le=1e-1,
        description="Weight decay (Optuna range: [1e-5, 1e-1], log-uniform)"
    )


    minority_weight_multiplier: float = Field(
        ...,
        ge=1.0,
        le=10.0,
        description="Multiplier for minority class (Optuna range: [1, 10])"
    )





class HyperparametersConfig(BaseModel):
    default_hypms: bool = Field(
        default=False,
        description="Whether to use default hyperparameters or site-specific ones"
    )
    site_hypms: SiteHyperparameters

class SiteConfig(BaseModel):
    name: str

    @validator("name")
    def validate_site(cls, v):
        allowed = ["cordoba", "hannover", "stockholm", "lyon", "belgrado", "marrakech"]
        if v not in allowed:
            raise ValueError(f"Invalid site '{v}', must be one of: {allowed}")
        return v


# Optional: if you want strict allowed names
ERA5_ALLOWED = {"g500", "g200", "psl"}
ERA5LAND_ALLOWED = {"swvl1", "swvl2", "swvl3"}


class DatasetConfig(BaseModel):
    variables_era5: List[str] = Field(..., min_items=1)
    variables_era5land: List[str] = Field(..., min_items=1)

    start_date_train: date
    end_date_train: date
    start_date_test: date
    end_date_test: date

    months: List[int] = Field(..., min_items=1)
    start_lag: int = Field(..., ge=0)
    lags_era5: int = Field(..., ge=0)

    use_spei: bool = Field(default=False, description="Whether to use spei or not")
    spei_spi: str
    scales_spei: List[str]
    distribution: str



    # Validate variable names
    @validator("variables_era5")
    def validate_era5_vars(cls, v):
        invalid = [x for x in v if x not in ERA5_ALLOWED]
        if invalid:
            raise ValueError(
                f"Invalid ERA5 variables: {invalid}. Allowed: {sorted(ERA5_ALLOWED)}"
            )
        return v

    @validator("variables_era5land")
    def validate_era5land_vars(cls, v):
        invalid = [x for x in v if x not in ERA5LAND_ALLOWED]
        if invalid:
            raise ValueError(
                f"Invalid ERA5Land variables: {invalid}. Allowed: {sorted(ERA5LAND_ALLOWED)}"
            )
        return v

    # Validate month list
    @validator("months")
    def validate_months(cls, v):
        for m in v:
            if not (1 <= m <= 12):
                raise ValueError(f"Month {m} is invalid — must be between 1 and 12")
        return v

    # Validate dates
    @validator("end_date_train")
    def validate_train_dates(cls, v, values):
        if "start_date_train" in values:
            if v < values["start_date_train"]:
                raise ValueError(
                    "end_date_train must be after start_date_train"
                )
        return v

    @validator("start_date_test")
    def validate_test_dates_start(cls, v, values):
        if "end_date_train" in values:
            if v <= values["end_date_train"]:
                raise ValueError(
                    "start_date_test must be *after* end_date_train"
                )
        return v

    @validator("end_date_test")
    def validate_test_dates_end(cls, v, values):
        if "start_date_test" in values:
            if v < values["start_date_test"]:
                raise ValueError(
                    "end_date_test must be after start_date_test"
                )
        return v

class EpochConfig(BaseModel):
    epochs: int = Field(gt=0, description="Number of training epochs")


class DatasetSchema(BaseModel):
    seed: int
    site: SiteConfig
    percentile: Literal["90p", "80p", "95p"]
    hyperparameters: HyperparametersConfig
    dataset: DatasetConfig
    epoch_config: EpochConfig
    paths: PathsConfig



def validate_schema(cfg: DictConfig) -> DatasetSchema:
    """Convert Hydra config → Pydantic Schema and validate."""
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    return DatasetSchema(**cfg_dict)
